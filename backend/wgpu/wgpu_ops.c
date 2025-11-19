#include "wgpu_backend.h"
#include "../../core/include/ops.h"
#include "../../core/include/registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Forward declaration for CPU implementation (not through registry to avoid circular dependency)
static Tensor* cpu_tensor_add(Tensor *A, Tensor *B);

// ====================================================
// Pipeline Cache
// ====================================================

typedef struct {
    WGPUShaderModule shader;
    WGPUBindGroupLayout bind_group_layout;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline;
    int initialized;
} PipelineCache;

static PipelineCache add_pipeline_cache = {0};
static PipelineCache matmul_pipeline_cache = {0};
static PipelineCache relu_pipeline_cache = {0};
static PipelineCache sigmoid_pipeline_cache = {0};
static PipelineCache tanh_pipeline_cache = {0};

// ====================================================
// Shader Source (embedded)
// ====================================================

// Optimized shader - removed heavy loops now that GPU is verified working
static const char* add_shader_source = 
    "@group(0) @binding(0) var<storage, read> input_a: array<f32>;\n"
    "@group(0) @binding(1) var<storage, read> input_b: array<f32>;\n"
    "@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n"
    "\n"
    "@compute @workgroup_size(256)\n"
    "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n"
    "    let threads_per_row = 65535u * 256u;\n"
    "    let index = global_id.y * threads_per_row + global_id.x;\n"
    "    if (index < arrayLength(&output)) {\n"
    "        output[index] = input_a[index] + input_b[index];\n"
    "    }\n"
    "}\n";

// Matrix multiplication shader with 16×16 tiling
static const char* matmul_shader_source =
    "@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;\n"
    "@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;\n"
    "@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;\n"
    "@group(0) @binding(3) var<uniform> dims: vec3<u32>;\n"
    "\n"
    "var<workgroup> tile_a: array<f32, 256>;\n"
    "var<workgroup> tile_b: array<f32, 256>;\n"
    "\n"
    "@compute @workgroup_size(16, 16)\n"
    "fn main(\n"
    "    @builtin(global_invocation_id) global_id: vec3<u32>,\n"
    "    @builtin(local_invocation_id) local_id: vec3<u32>\n"
    ") {\n"
    "    let M = dims.x;\n"
    "    let K = dims.y;\n"
    "    let N = dims.z;\n"
    "    let row = global_id.y;\n"
    "    let col = global_id.x;\n"
    "    let local_row = local_id.y;\n"
    "    let local_col = local_id.x;\n"
    "    var sum: f32 = 0.0;\n"
    "    let num_tiles = (K + 15u) / 16u;\n"
    "    for (var tile: u32 = 0u; tile < num_tiles; tile++) {\n"
    "        let a_row = row;\n"
    "        let a_col = tile * 16u + local_col;\n"
    "        if (a_row < M && a_col < K) {\n"
    "            tile_a[local_row * 16u + local_col] = matrix_a[a_row * K + a_col];\n"
    "        } else {\n"
    "            tile_a[local_row * 16u + local_col] = 0.0;\n"
    "        }\n"
    "        let b_row = tile * 16u + local_row;\n"
    "        let b_col = col;\n"
    "        if (b_row < K && b_col < N) {\n"
    "            tile_b[local_row * 16u + local_col] = matrix_b[b_row * N + b_col];\n"
    "        } else {\n"
    "            tile_b[local_row * 16u + local_col] = 0.0;\n"
    "        }\n"
    "        workgroupBarrier();\n"
    "        for (var k: u32 = 0u; k < 16u; k++) {\n"
    "            sum += tile_a[local_row * 16u + k] * tile_b[k * 16u + local_col];\n"
    "        }\n"
    "        workgroupBarrier();\n"
    "    }\n"
    "    if (row < M && col < N) {\n"
    "        matrix_c[row * N + col] = sum;\n"
    "    }\n"
    "}\n";

// ====================================================
// GPU Operations
// ====================================================

Tensor* wgpu_tensor_add(Tensor *A, Tensor *B) {
    if (!wgpu_available() || !A || !B) {
        return cpu_tensor_add(A, B);  // CPU fallback
    }
    
    // Validate dimensions match
    if (A->ndim != B->ndim) {
        fprintf(stderr, "[WGPU] Dimension mismatch in add: %d vs %d\n", A->ndim, B->ndim);
        return cpu_tensor_add(A, B);  // CPU fallback
    }
    
    for (int i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i]) {
            fprintf(stderr, "[WGPU] Shape mismatch in add at dim %d: %d vs %d\n", 
                    i, A->shape[i], B->shape[i]);
            return cpu_tensor_add(A, B);  // CPU fallback
        }
    }
    
    size_t size = A->size;
    size_t buffer_size = size * sizeof(float);
    
    // Create buffers
    WGPUBuffer buffer_a = wgpu_create_buffer(buffer_size, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_b = wgpu_create_buffer(buffer_size, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_out = wgpu_create_buffer(buffer_size, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    
    // Create staging buffer for reading results
    WGPUBuffer staging_buffer = wgpu_create_buffer(buffer_size,
        WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    
    if (!buffer_a || !buffer_b || !buffer_out || !staging_buffer) {
        fprintf(stderr, "[WGPU] Failed to create buffers for add\n");
        if (buffer_a) wgpuBufferRelease(buffer_a);
        if (buffer_b) wgpuBufferRelease(buffer_b);
        if (buffer_out) wgpuBufferRelease(buffer_out);
        if (staging_buffer) wgpuBufferRelease(staging_buffer);
        return cpu_tensor_add(A, B);  // CPU fallback
    }
    
    // Upload data
    wgpu_write_buffer(buffer_a, 0, A->data, buffer_size);
    wgpu_write_buffer(buffer_b, 0, B->data, buffer_size);
    
    // Initialize pipeline cache on first use
    if (!add_pipeline_cache.initialized) {
        // Create shader module
        WGPUShaderModuleWGSLDescriptor wgsl_desc = {
            .chain = {.sType = WGPUSType_ShaderModuleWGSLDescriptor},
            .code = add_shader_source,
        };
        WGPUShaderModuleDescriptor shader_desc = {
            .nextInChain = (const WGPUChainedStruct*)&wgsl_desc,
            .label = "add_shader",
        };
        add_pipeline_cache.shader = wgpuDeviceCreateShaderModule(wgpu_get_device(), &shader_desc);
        
        if (!add_pipeline_cache.shader) {
            fprintf(stderr, "[WGPU] Failed to create shader module\n");
            wgpuBufferRelease(buffer_a);
            wgpuBufferRelease(buffer_b);
            wgpuBufferRelease(buffer_out);
            wgpuBufferRelease(staging_buffer);
            return cpu_tensor_add(A, B);
        }
        
        // Create bind group layout
        WGPUBindGroupLayoutEntry layout_entries[3] = {
            {
                .binding = 0,
                .visibility = WGPUShaderStage_Compute,
                .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage},
            },
            {
                .binding = 1,
                .visibility = WGPUShaderStage_Compute,
                .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage},
            },
            {
                .binding = 2,
                .visibility = WGPUShaderStage_Compute,
                .buffer = {.type = WGPUBufferBindingType_Storage},
            },
        };
        
        WGPUBindGroupLayoutDescriptor bgl_desc = {
            .entryCount = 3,
            .entries = layout_entries,
        };
        add_pipeline_cache.bind_group_layout = wgpuDeviceCreateBindGroupLayout(wgpu_get_device(), &bgl_desc);
        
        // Create pipeline layout
        WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
            .bindGroupLayoutCount = 1,
            .bindGroupLayouts = &add_pipeline_cache.bind_group_layout,
        };
        add_pipeline_cache.pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_get_device(), &pipeline_layout_desc);
        
        // Create compute pipeline
        WGPUComputePipelineDescriptor pipeline_desc = {
            .layout = add_pipeline_cache.pipeline_layout,
            .compute = {
                .module = add_pipeline_cache.shader,
                .entryPoint = "main",
            },
        };
        add_pipeline_cache.pipeline = wgpuDeviceCreateComputePipeline(wgpu_get_device(), &pipeline_desc);
        add_pipeline_cache.initialized = 1;
    }
    
    // Create bind group (per-operation, uses cached layout)
    WGPUBindGroupEntry bind_entries[3] = {
        {.binding = 0, .buffer = buffer_a, .size = buffer_size},
        {.binding = 1, .buffer = buffer_b, .size = buffer_size},
        {.binding = 2, .buffer = buffer_out, .size = buffer_size},
    };
    
    WGPUBindGroupDescriptor bind_group_desc = {
        .layout = add_pipeline_cache.bind_group_layout,
        .entryCount = 3,
        .entries = bind_entries,
    };
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(wgpu_get_device(), &bind_group_desc);
    
    // Create command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(wgpu_get_device(), &encoder_desc);
    
    // Encode compute pass (uses cached pipeline)
    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
    wgpuComputePassEncoderSetPipeline(pass, add_pipeline_cache.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    
    // Dispatch workgroups (256 threads per workgroup)
    // WebGPU limit: max 65535 per dimension, use 2D dispatch for large tensors
    uint32_t total_workgroups = (size + 255) / 256;
    uint32_t workgroups_x, workgroups_y;
    
    if (total_workgroups <= 65535) {
        workgroups_x = total_workgroups;
        workgroups_y = 1;
    } else {
        // Split into 2D grid: x * y >= total_workgroups, both <= 65535
        workgroups_x = 65535;
        workgroups_y = (total_workgroups + 65534) / 65535;
    }
    
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups_x, workgroups_y, 1);
    wgpuComputePassEncoderEnd(pass);
    
    // Copy output buffer to staging buffer for CPU readback
    wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer_out, 0, staging_buffer, 0, buffer_size);
    
    // Submit commands
    WGPUCommandBufferDescriptor cmd_buffer_desc = {0};
    WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, &cmd_buffer_desc);
    wgpuQueueSubmit(wgpu_get_queue(), 1, &cmd_buffer);
    
    // Wait for GPU work to complete
    for (int i = 0; i < 10000; i++) {
        wgpuDevicePoll(wgpu_get_device(), 0, NULL);
        for (volatile int j = 0; j < 1000; j++);
    }
    
    // Create output tensor
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) {
        fprintf(stderr, "[WGPU] Failed to create output tensor\n");
        wgpuBufferRelease(buffer_a);
        wgpuBufferRelease(buffer_b);
        wgpuBufferRelease(buffer_out);
        wgpuBufferRelease(staging_buffer);
        wgpuBindGroupRelease(bind_group);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(encoder);
        wgpuCommandBufferRelease(cmd_buffer);
        return cpu_tensor_add(A, B);
    }
    
    // Read back results from staging buffer
    if (wgpu_read_buffer(staging_buffer, 0, C->data, buffer_size) != 0) {
        fprintf(stderr, "[WGPU] Failed to read back results\n");
        tensor_free(C);
        C = cpu_tensor_add(A, B);  // CPU fallback
    }
    
    // Cleanup per-operation GPU resources (pipeline is cached, don't release)
    wgpuBufferRelease(buffer_a);
    wgpuBufferRelease(buffer_b);
    wgpuBufferRelease(buffer_out);
    wgpuBufferRelease(staging_buffer);
    wgpuBindGroupRelease(bind_group);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuCommandBufferRelease(cmd_buffer);
    
    // Setup autograd info (same as CPU version)
    if (A->requires_grad || B->requires_grad) {
        C->requires_grad = 1;
        C->op_name = "add";
        C->num_inputs = 2;
        C->inputs = malloc(2 * sizeof(Tensor*));
        C->inputs[0] = A;
        C->inputs[1] = B;
        
        // Set the backward function (registered in registry)
        BackwardFn backward_fn = get_tensor_op_backward_fn("add");
        if (backward_fn) {
            C->backward_fn = backward_fn;
        }
    }
    
    return C;
}

static Tensor* cpu_tensor_matmul(Tensor *A, Tensor *B) {
    if (!A || !B) return NULL;
    
    // Only implement 2D × 2D here for GPU fallback
    if (A->ndim != 2 || B->ndim != 2) {
        fprintf(stderr, "[CPU] Unsupported matmul dimensions in fallback\n");
        return NULL;
    }
    
    size_t M = A->shape[0];
    size_t K = A->shape[1];
    size_t N = B->shape[1];
    
    if (B->shape[0] != K) return NULL;
    
    size_t C_shape[2] = {M, N};
    Tensor *C = tensor_create(C_shape, 2);
    if (!C) return NULL;
    
    // Standard triple-nested loop
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += A->data[i * K + k] * B->data[k * N + j];
            }
            C->data[i * N + j] = acc;
        }
    }
    
    // Setup autograd
    if (A->requires_grad || B->requires_grad) {
        C->requires_grad = 1;
        C->op_name = "matmul";
        C->num_inputs = 2;
        C->inputs = malloc(2 * sizeof(Tensor*));
        C->inputs[0] = A;
        C->inputs[1] = B;
        
        BackwardFn backward_fn = get_tensor_op_backward_fn("matmul");
        if (backward_fn) {
            C->backward_fn = backward_fn;
        }
    }
    
    return C;
}

// Forward declaration for CPU matmul
static Tensor* cpu_tensor_matmul(Tensor *A, Tensor *B);

Tensor* wgpu_tensor_matmul(Tensor *A, Tensor *B) {
    if (!wgpu_available() || !A || !B) {
        return cpu_tensor_matmul(A, B);
    }
    
    // Only handle 2D × 2D case on GPU for now
    if (A->ndim != 2 || B->ndim != 2) {
        return cpu_tensor_matmul(A, B);
    }
    
    size_t M = A->shape[0];
    size_t K = A->shape[1];
    size_t N = B->shape[1];
    
    // Validate dimensions
    if (B->shape[0] != K) {
        fprintf(stderr, "[WGPU] Matrix dimension mismatch: A(%zu×%zu) × B(%zu×%zu)\n",
                M, K, B->shape[0], N);
        return cpu_tensor_matmul(A, B);
    }
    
    // Create buffers
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    
    WGPUBuffer buffer_a = wgpu_create_buffer(size_a, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_b = wgpu_create_buffer(size_b, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_c = wgpu_create_buffer(size_c, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    
    // Create uniform buffer for dimensions
    uint32_t dims[3] = {(uint32_t)M, (uint32_t)K, (uint32_t)N};
    WGPUBuffer uniform_buffer = wgpu_create_buffer(sizeof(dims),
        WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    
    // Staging buffer for result readback
    WGPUBuffer staging_buffer = wgpu_create_buffer(size_c,
        WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    
    if (!buffer_a || !buffer_b || !buffer_c || !uniform_buffer || !staging_buffer) {
        fprintf(stderr, "[WGPU] Failed to create buffers for matmul\n");
        if (buffer_a) wgpuBufferRelease(buffer_a);
        if (buffer_b) wgpuBufferRelease(buffer_b);
        if (buffer_c) wgpuBufferRelease(buffer_c);
        if (uniform_buffer) wgpuBufferRelease(uniform_buffer);
        if (staging_buffer) wgpuBufferRelease(staging_buffer);
        return cpu_tensor_matmul(A, B);
    }
    
    // Upload data
    wgpu_write_buffer(buffer_a, 0, A->data, size_a);
    wgpu_write_buffer(buffer_b, 0, B->data, size_b);
    wgpu_write_buffer(uniform_buffer, 0, dims, sizeof(dims));
    
    // Initialize pipeline cache on first use
    if (!matmul_pipeline_cache.initialized) {
        WGPUShaderModuleWGSLDescriptor wgsl_desc = {
            .chain = {.sType = WGPUSType_ShaderModuleWGSLDescriptor},
            .code = matmul_shader_source,
        };
        WGPUShaderModuleDescriptor shader_desc = {
            .nextInChain = (const WGPUChainedStruct*)&wgsl_desc,
            .label = "matmul_shader",
        };
        matmul_pipeline_cache.shader = wgpuDeviceCreateShaderModule(wgpu_get_device(), &shader_desc);
        
        if (!matmul_pipeline_cache.shader) {
            fprintf(stderr, "[WGPU] Failed to create matmul shader module\n");
            wgpuBufferRelease(buffer_a);
            wgpuBufferRelease(buffer_b);
            wgpuBufferRelease(buffer_c);
            wgpuBufferRelease(uniform_buffer);
            wgpuBufferRelease(staging_buffer);
            return cpu_tensor_matmul(A, B);
        }
        
        // Create bind group layout (3 storage + 1 uniform)
        WGPUBindGroupLayoutEntry layout_entries[4] = {
            {
                .binding = 0,
                .visibility = WGPUShaderStage_Compute,
                .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage},
            },
            {
                .binding = 1,
                .visibility = WGPUShaderStage_Compute,
                .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage},
            },
            {
                .binding = 2,
                .visibility = WGPUShaderStage_Compute,
                .buffer = {.type = WGPUBufferBindingType_Storage},
            },
            {
                .binding = 3,
                .visibility = WGPUShaderStage_Compute,
                .buffer = {.type = WGPUBufferBindingType_Uniform},
            },
        };
        
        WGPUBindGroupLayoutDescriptor bgl_desc = {
            .entryCount = 4,
            .entries = layout_entries,
        };
        matmul_pipeline_cache.bind_group_layout = wgpuDeviceCreateBindGroupLayout(wgpu_get_device(), &bgl_desc);
        
        WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
            .bindGroupLayoutCount = 1,
            .bindGroupLayouts = &matmul_pipeline_cache.bind_group_layout,
        };
        matmul_pipeline_cache.pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_get_device(), &pipeline_layout_desc);
        
        WGPUComputePipelineDescriptor pipeline_desc = {
            .layout = matmul_pipeline_cache.pipeline_layout,
            .compute = {
                .module = matmul_pipeline_cache.shader,
                .entryPoint = "main",
            },
        };
        matmul_pipeline_cache.pipeline = wgpuDeviceCreateComputePipeline(wgpu_get_device(), &pipeline_desc);
        matmul_pipeline_cache.initialized = 1;
    }
    
    // Create bind group
    WGPUBindGroupEntry bind_entries[4] = {
        {.binding = 0, .buffer = buffer_a, .size = size_a},
        {.binding = 1, .buffer = buffer_b, .size = size_b},
        {.binding = 2, .buffer = buffer_c, .size = size_c},
        {.binding = 3, .buffer = uniform_buffer, .size = sizeof(dims)},
    };
    
    WGPUBindGroupDescriptor bind_group_desc = {
        .layout = matmul_pipeline_cache.bind_group_layout,
        .entryCount = 4,
        .entries = bind_entries,
    };
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(wgpu_get_device(), &bind_group_desc);
    
    // Create command encoder
    WGPUCommandEncoderDescriptor encoder_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(wgpu_get_device(), &encoder_desc);
    
    // Encode compute pass with 16×16 workgroups
    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
    wgpuComputePassEncoderSetPipeline(pass, matmul_pipeline_cache.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    
    // Dispatch: need (M+15)/16 × (N+15)/16 workgroups
    uint32_t workgroups_x = (N + 15) / 16;
    uint32_t workgroups_y = (M + 15) / 16;
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups_x, workgroups_y, 1);
    wgpuComputePassEncoderEnd(pass);
    
    // Copy result to staging buffer
    wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer_c, 0, staging_buffer, 0, size_c);
    
    // Submit
    WGPUCommandBufferDescriptor cmd_buffer_desc = {0};
    WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, &cmd_buffer_desc);
    wgpuQueueSubmit(wgpu_get_queue(), 1, &cmd_buffer);
    
    // Wait for completion
    for (int i = 0; i < 10000; i++) {
        wgpuDevicePoll(wgpu_get_device(), 0, NULL);
        for (volatile int j = 0; j < 1000; j++);
    }
    
    // Create output tensor
    size_t C_shape[2] = {M, N};
    Tensor *C = tensor_create(C_shape, 2);
    if (!C) {
        fprintf(stderr, "[WGPU] Failed to create output tensor\n");
        wgpuBufferRelease(buffer_a);
        wgpuBufferRelease(buffer_b);
        wgpuBufferRelease(buffer_c);
        wgpuBufferRelease(uniform_buffer);
        wgpuBufferRelease(staging_buffer);
        wgpuBindGroupRelease(bind_group);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(encoder);
        wgpuCommandBufferRelease(cmd_buffer);
        return cpu_tensor_matmul(A, B);
    }
    
    // Read back results
    if (wgpu_read_buffer(staging_buffer, 0, C->data, size_c) != 0) {
        fprintf(stderr, "[WGPU] Failed to read back matmul results\n");
        tensor_free(C);
        C = cpu_tensor_matmul(A, B);
    }
    
    // Cleanup
    wgpuBufferRelease(buffer_a);
    wgpuBufferRelease(buffer_b);
    wgpuBufferRelease(buffer_c);
    wgpuBufferRelease(uniform_buffer);
    wgpuBufferRelease(staging_buffer);
    wgpuBindGroupRelease(bind_group);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuCommandBufferRelease(cmd_buffer);
    
    // Setup autograd
    if (A->requires_grad || B->requires_grad) {
        C->requires_grad = 1;
        C->op_name = "matmul";
        C->num_inputs = 2;
        C->inputs = malloc(2 * sizeof(Tensor*));
        C->inputs[0] = A;
        C->inputs[1] = B;
        
        BackwardFn backward_fn = get_tensor_op_backward_fn("matmul");
        if (backward_fn) {
            C->backward_fn = backward_fn;
        }
    }
    
    return C;
}

// ====================================================
// Activation Functions (element-wise, simple shaders)
// ====================================================

// Shared unary operation shader template (ReLU, Sigmoid, Tanh)
static Tensor* wgpu_unary_op(Tensor *input, const char *op_name, const char *shader_code) {
    if (!wgpu_available() || !input) return NULL;
    
    size_t size = input->size;
    size_t buffer_size = size * sizeof(float);
    
    WGPUBuffer buffer_in = wgpu_create_buffer(buffer_size, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_out = wgpu_create_buffer(buffer_size, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer staging_buffer = wgpu_create_buffer(buffer_size,
        WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    
    if (!buffer_in || !buffer_out || !staging_buffer) {
        if (buffer_in) wgpuBufferRelease(buffer_in);
        if (buffer_out) wgpuBufferRelease(buffer_out);
        if (staging_buffer) wgpuBufferRelease(staging_buffer);
        return NULL;
    }
    
    wgpu_write_buffer(buffer_in, 0, input->data, buffer_size);
    
    // Create shader module
    WGPUShaderModuleWGSLDescriptor wgsl_desc = {
        .chain = {.sType = WGPUSType_ShaderModuleWGSLDescriptor},
        .code = shader_code,
    };
    WGPUShaderModuleDescriptor shader_desc = {
        .nextInChain = (const WGPUChainedStruct*)&wgsl_desc,
    };
    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(wgpu_get_device(), &shader_desc);
    
    // Create bind group layout
    WGPUBindGroupLayoutEntry layout_entries[2] = {
        {.binding = 0, .visibility = WGPUShaderStage_Compute, 
         .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
        {.binding = 1, .visibility = WGPUShaderStage_Compute, 
         .buffer = {.type = WGPUBufferBindingType_Storage}},
    };
    WGPUBindGroupLayoutDescriptor bgl_desc = {.entryCount = 2, .entries = layout_entries};
    WGPUBindGroupLayout bind_group_layout = wgpuDeviceCreateBindGroupLayout(wgpu_get_device(), &bgl_desc);
    
    // Create pipeline
    WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
        .bindGroupLayoutCount = 1, .bindGroupLayouts = &bind_group_layout};
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_get_device(), &pipeline_layout_desc);
    
    WGPUComputePipelineDescriptor pipeline_desc = {
        .layout = pipeline_layout,
        .compute = {.module = shader, .entryPoint = "main"},
    };
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(wgpu_get_device(), &pipeline_desc);
    
    // Create bind group
    WGPUBindGroupEntry bind_entries[2] = {
        {.binding = 0, .buffer = buffer_in, .size = buffer_size},
        {.binding = 1, .buffer = buffer_out, .size = buffer_size},
    };
    WGPUBindGroupDescriptor bind_group_desc = {
        .layout = bind_group_layout, .entryCount = 2, .entries = bind_entries};
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(wgpu_get_device(), &bind_group_desc);
    
    // Encode and dispatch
    WGPUCommandEncoderDescriptor encoder_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(wgpu_get_device(), &encoder_desc);
    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
    
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    
    uint32_t total_workgroups = (size + 255) / 256;
    uint32_t workgroups_x = total_workgroups <= 65535 ? total_workgroups : 65535;
    uint32_t workgroups_y = total_workgroups <= 65535 ? 1 : (total_workgroups + 65534) / 65535;
    
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups_x, workgroups_y, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer_out, 0, staging_buffer, 0, buffer_size);
    
    WGPUCommandBufferDescriptor cmd_buffer_desc = {0};
    WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, &cmd_buffer_desc);
    wgpuQueueSubmit(wgpu_get_queue(), 1, &cmd_buffer);
    
    // Poll for completion
    for (int i = 0; i < 10000; i++) {
        wgpuDevicePoll(wgpu_get_device(), 0, NULL);
        for (volatile int j = 0; j < 1000; j++);
    }
    
    // Read results
    Tensor *output = tensor_create(input->shape, input->ndim);
    if (output && wgpu_read_buffer(staging_buffer, 0, output->data, buffer_size) == 0) {
        // Setup autograd
        if (input->requires_grad) {
            output->requires_grad = 1;
            output->op_name = op_name;
            output->num_inputs = 1;
            output->inputs = malloc(sizeof(Tensor*));
            output->inputs[0] = input;
            output->backward_fn = get_tensor_op_backward_fn(op_name);
        }
    }
    
    // Cleanup
    wgpuBufferRelease(buffer_in);
    wgpuBufferRelease(buffer_out);
    wgpuBufferRelease(staging_buffer);
    wgpuBindGroupRelease(bind_group);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuCommandBufferRelease(cmd_buffer);
    wgpuComputePipelineRelease(pipeline);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuBindGroupLayoutRelease(bind_group_layout);
    wgpuShaderModuleRelease(shader);
    
    return output;
}

// ReLU with cached pipeline
Tensor* wgpu_tensor_relu(Tensor *input) {
    if (!wgpu_available() || !input) return NULL;
    
    size_t size = input->size;
    
    // For small tensors (< 1M elements), GPU overhead exceeds benefit - use CPU
    if (size < 1000000) return NULL;
    size_t buffer_size = size * sizeof(float);
    
    // Create buffers
    WGPUBuffer buffer_in = wgpu_create_buffer(buffer_size, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer buffer_out = wgpu_create_buffer(buffer_size, 
        WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer staging_buffer = wgpu_create_buffer(buffer_size,
        WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    
    if (!buffer_in || !buffer_out || !staging_buffer) {
        if (buffer_in) wgpuBufferRelease(buffer_in);
        if (buffer_out) wgpuBufferRelease(buffer_out);
        if (staging_buffer) wgpuBufferRelease(staging_buffer);
        return NULL;
    }
    
    wgpu_write_buffer(buffer_in, 0, input->data, buffer_size);
    
    // Initialize pipeline cache on first use
    if (!relu_pipeline_cache.initialized) {
        static const char* shader_code = 
            "@group(0) @binding(0) var<storage, read> input: array<f32>;\n"
            "@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n"
            "@compute @workgroup_size(256)\n"
            "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n"
            "    let threads_per_row = 65535u * 256u;\n"
            "    let index = global_id.y * threads_per_row + global_id.x;\n"
            "    if (index < arrayLength(&output)) {\n"
            "        output[index] = max(input[index], 0.0);\n"
            "    }\n"
            "}\n";
        
        WGPUShaderModuleWGSLDescriptor wgsl_desc = {
            .chain = {.sType = WGPUSType_ShaderModuleWGSLDescriptor},
            .code = shader_code,
        };
        WGPUShaderModuleDescriptor shader_desc = {
            .nextInChain = (const WGPUChainedStruct*)&wgsl_desc,
        };
        relu_pipeline_cache.shader = wgpuDeviceCreateShaderModule(wgpu_get_device(), &shader_desc);
        
        WGPUBindGroupLayoutEntry layout_entries[2] = {
            {.binding = 0, .visibility = WGPUShaderStage_Compute, 
             .buffer = {.type = WGPUBufferBindingType_ReadOnlyStorage}},
            {.binding = 1, .visibility = WGPUShaderStage_Compute, 
             .buffer = {.type = WGPUBufferBindingType_Storage}},
        };
        WGPUBindGroupLayoutDescriptor bgl_desc = {.entryCount = 2, .entries = layout_entries};
        relu_pipeline_cache.bind_group_layout = wgpuDeviceCreateBindGroupLayout(wgpu_get_device(), &bgl_desc);
        
        WGPUPipelineLayoutDescriptor pipeline_layout_desc = {
            .bindGroupLayoutCount = 1, .bindGroupLayouts = &relu_pipeline_cache.bind_group_layout};
        relu_pipeline_cache.pipeline_layout = wgpuDeviceCreatePipelineLayout(wgpu_get_device(), &pipeline_layout_desc);
        
        WGPUComputePipelineDescriptor pipeline_desc = {
            .layout = relu_pipeline_cache.pipeline_layout,
            .compute = {.module = relu_pipeline_cache.shader, .entryPoint = "main"},
        };
        relu_pipeline_cache.pipeline = wgpuDeviceCreateComputePipeline(wgpu_get_device(), &pipeline_desc);
        relu_pipeline_cache.initialized = 1;
    }
    
    // Create bind group (per-operation)
    WGPUBindGroupEntry bind_entries[2] = {
        {.binding = 0, .buffer = buffer_in, .size = buffer_size},
        {.binding = 1, .buffer = buffer_out, .size = buffer_size},
    };
    WGPUBindGroupDescriptor bind_group_desc = {
        .layout = relu_pipeline_cache.bind_group_layout, .entryCount = 2, .entries = bind_entries};
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(wgpu_get_device(), &bind_group_desc);
    
    // Encode and dispatch
    WGPUCommandEncoderDescriptor encoder_desc = {0};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(wgpu_get_device(), &encoder_desc);
    WGPUComputePassDescriptor pass_desc = {0};
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
    
    wgpuComputePassEncoderSetPipeline(pass, relu_pipeline_cache.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    
    uint32_t total_workgroups = (size + 255) / 256;
    uint32_t workgroups_x = total_workgroups <= 65535 ? total_workgroups : 65535;
    uint32_t workgroups_y = total_workgroups <= 65535 ? 1 : (total_workgroups + 65534) / 65535;
    
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups_x, workgroups_y, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer_out, 0, staging_buffer, 0, buffer_size);
    
    WGPUCommandBufferDescriptor cmd_buffer_desc = {0};
    WGPUCommandBuffer cmd_buffer = wgpuCommandEncoderFinish(encoder, &cmd_buffer_desc);
    wgpuQueueSubmit(wgpu_get_queue(), 1, &cmd_buffer);
    
    // Poll for completion
    for (int i = 0; i < 10000; i++) {
        wgpuDevicePoll(wgpu_get_device(), 0, NULL);
        for (volatile int j = 0; j < 1000; j++);
    }
    
    // Read results
    Tensor *output = tensor_create(input->shape, input->ndim);
    if (output && wgpu_read_buffer(staging_buffer, 0, output->data, buffer_size) == 0) {
        if (input->requires_grad) {
            output->requires_grad = 1;
            output->op_name = "relu";
            output->num_inputs = 1;
            output->inputs = malloc(sizeof(Tensor*));
            output->inputs[0] = input;
            output->backward_fn = get_tensor_op_backward_fn("relu");
        }
    }
    
    // Cleanup
    wgpuBufferRelease(buffer_in);
    wgpuBufferRelease(buffer_out);
    wgpuBufferRelease(staging_buffer);
    wgpuBindGroupRelease(bind_group);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuCommandBufferRelease(cmd_buffer);
    
    return output;
}

// Sigmoid and Tanh - keep simple for now, can be optimized later
Tensor* wgpu_tensor_sigmoid(Tensor *input) {
    // For now, use CPU - activation functions are bandwidth-bound anyway
    return NULL;
}

Tensor* wgpu_tensor_tanh(Tensor *input) {
    // For now, use CPU - activation functions are bandwidth-bound anyway
    return NULL;
}

Tensor* wgpu_tensor_softmax(Tensor *input) {
    // Softmax requires row-wise operations, keep on CPU for now
    // TODO: Implement efficient GPU softmax with workgroup reductions
    return NULL;  // Falls back to CPU
}

// ====================================================
// Pipeline Cache Cleanup
// ====================================================

void wgpu_cleanup_pipeline_caches(void) {
    if (add_pipeline_cache.initialized) {
        wgpuComputePipelineRelease(add_pipeline_cache.pipeline);
        wgpuPipelineLayoutRelease(add_pipeline_cache.pipeline_layout);
        wgpuBindGroupLayoutRelease(add_pipeline_cache.bind_group_layout);
        wgpuShaderModuleRelease(add_pipeline_cache.shader);
        add_pipeline_cache.initialized = 0;
    }
    
    if (matmul_pipeline_cache.initialized) {
        wgpuComputePipelineRelease(matmul_pipeline_cache.pipeline);
        wgpuPipelineLayoutRelease(matmul_pipeline_cache.pipeline_layout);
        wgpuBindGroupLayoutRelease(matmul_pipeline_cache.bind_group_layout);
        wgpuShaderModuleRelease(matmul_pipeline_cache.shader);
        matmul_pipeline_cache.initialized = 0;
    }
    
    if (relu_pipeline_cache.initialized) {
        wgpuComputePipelineRelease(relu_pipeline_cache.pipeline);
        wgpuPipelineLayoutRelease(relu_pipeline_cache.pipeline_layout);
        wgpuBindGroupLayoutRelease(relu_pipeline_cache.bind_group_layout);
        wgpuShaderModuleRelease(relu_pipeline_cache.shader);
        relu_pipeline_cache.initialized = 0;
    }
}

// ====================================================
// Registration
// ====================================================

void wgpu_register_ops(void) {
    if (!wgpu_available()) {
        return;
    }
    
    // Register GPU implementations with high priority
    register_operation_backend("add", (void*)wgpu_tensor_add, 10);
    register_operation_backend("matmul", (void*)wgpu_tensor_matmul, 10);
    register_operation_backend("relu", (void*)wgpu_tensor_relu, 10);
    register_operation_backend("sigmoid", (void*)wgpu_tensor_sigmoid, 10);
    register_operation_backend("tanh", (void*)wgpu_tensor_tanh, 10);
    register_operation_backend("softmax", (void*)wgpu_tensor_softmax, 10);
}

// ====================================================
// CPU Fallback Implementation
// ====================================================

static Tensor* cpu_tensor_add(Tensor *A, Tensor *B) {
    if (!A || !B) return NULL;
    
    // Handle broadcasting case (2D + 1D bias)
    if (A->ndim == 2 && B->ndim == 1 && A->shape[1] == B->shape[0]) {
        Tensor *C = tensor_create(A->shape, A->ndim);
        if (!C) return NULL;
        
        for (size_t i = 0; i < A->shape[0]; i++) {
            for (size_t j = 0; j < A->shape[1]; j++) {
                C->data[i * A->shape[1] + j] = A->data[i * A->shape[1] + j] + B->data[j];
            }
        }
        
        // Setup autograd
        if (A->requires_grad || B->requires_grad) {
            C->requires_grad = 1;
            C->op_name = "add";
            C->num_inputs = 2;
            C->inputs = malloc(2 * sizeof(Tensor*));
            C->inputs[0] = A;
            C->inputs[1] = B;
            
            BackwardFn backward_fn = get_tensor_op_backward_fn("add");
            if (backward_fn) {
                C->backward_fn = backward_fn;
            }
        }
        
        return C;
    }
    
    // Same-shape element-wise addition
    if (A->ndim != B->ndim) return NULL;
    for (int i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i]) return NULL;
    }
    
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;
    
    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = A->data[i] + B->data[i];
    }
    
    // Setup autograd
    if (A->requires_grad || B->requires_grad) {
        C->requires_grad = 1;
        C->op_name = "add";
        C->num_inputs = 2;
        C->inputs = malloc(2 * sizeof(Tensor*));
        C->inputs[0] = A;
        C->inputs[1] = B;
        
        BackwardFn backward_fn = get_tensor_op_backward_fn("add");
        if (backward_fn) {
            C->backward_fn = backward_fn;
        }
    }
    
    return C;
}
