#include "../../include/basednn.h"
#include <stdio.h>
#include <sys/time.h>

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// CPU-only matmul for comparison
static Tensor* cpu_matmul(Tensor *A, Tensor *B) {
    if (!A || !B || A->ndim != 2 || B->ndim != 2) return NULL;
    
    size_t M = A->shape[0];
    size_t K = A->shape[1];
    size_t N = B->shape[1];
    
    if (B->shape[0] != K) return NULL;
    
    size_t C_shape[2] = {M, N};
    Tensor *C = tensor_create(C_shape, 2);
    if (!C) return NULL;
    
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += A->data[i * K + k] * B->data[k * N + j];
            }
            C->data[i * N + j] = acc;
        }
    }
    
    return C;
}

int main() {
    printf("=== Detailed GPU vs CPU Comparison for Neural Network ===\n\n");
    
    // First, test with GPU enabled
    printf("--- Test 1: With GPU Backend ---\n");
    basednn_init();
    
    // Create network layers
    Tensor *input = tensor_randn((size_t[]){64, 784}, 2, 42);
    Tensor *W1 = tensor_randn((size_t[]){784, 256}, 2, 43);
    Tensor *W2 = tensor_randn((size_t[]){256, 128}, 2, 44);
    
    // Time individual operations
    printf("\n1. Matrix Multiplication (64x784 @ 784x256):\n");
    
    // Warmup
    Tensor *warmup = tensor_matmul(input, W1);
    tensor_free(warmup);
    
    double start = get_time_ms();
    for (int i = 0; i < 5; i++) {
        Tensor *result = tensor_matmul(input, W1);
        tensor_free(result);
    }
    double gpu_matmul_time = (get_time_ms() - start) / 5.0;
    printf("   GPU (via registry): %.2f ms avg\n", gpu_matmul_time);
    
    // Now test CPU directly
    start = get_time_ms();
    for (int i = 0; i < 5; i++) {
        Tensor *result = cpu_matmul(input, W1);
        tensor_free(result);
    }
    double cpu_matmul_time = (get_time_ms() - start) / 5.0;
    printf("   CPU (direct):       %.2f ms avg\n", cpu_matmul_time);
    printf("   Speedup:            %.1fx\n", cpu_matmul_time / gpu_matmul_time);
    
    // Test ReLU
    printf("\n2. ReLU Activation (64x256 elements):\n");
    Tensor *relu_input = tensor_randn((size_t[]){64, 256}, 2, 45);
    
    start = get_time_ms();
    for (int i = 0; i < 20; i++) {
        Tensor *result = tensor_relu(relu_input);
        tensor_free(result);
    }
    double relu_time = (get_time_ms() - start) / 20.0;
    printf("   Time: %.2f ms avg\n", relu_time);
    
    // Test full forward pass
    printf("\n3. Full Forward Pass (784->256->128->10):\n");
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(784, 256)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(256, 128)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(128, 10)));
    
    Tensor *batch = tensor_randn((size_t[]){64, 784}, 2, 46);
    
    // Warmup
    Tensor *out_warmup = network_forward(net, batch);
    tensor_free(out_warmup);
    
    start = get_time_ms();
    for (int i = 0; i < 10; i++) {
        Tensor *output = network_forward(net, batch);
        tensor_free(output);
    }
    double forward_time = (get_time_ms() - start) / 10.0;
    printf("   Time: %.2f ms avg (batch of 64)\n", forward_time);
    printf("   Throughput: %.1f samples/sec\n", 640.0 / (forward_time / 1000.0));
    
    // Detailed breakdown
    printf("\n4. Operation Breakdown:\n");
    printf("   64x784 @ 784x256 matmul: %.2f ms\n", gpu_matmul_time);
    printf("   ReLU(64x256):            %.2f ms\n", relu_time);
    
    Tensor *h1 = tensor_matmul(batch, W2);
    start = get_time_ms();
    for (int i = 0; i < 10; i++) {
        Tensor *result = tensor_matmul(h1, W2);
        tensor_free(result);
    }
    double matmul2_time = (get_time_ms() - start) / 10.0;
    printf("   64x256 @ 256x128 matmul: %.2f ms\n", matmul2_time);
    
    printf("\n5. GPU Utilization Check:\n");
    printf("   If GPU is being used, large matmuls should be 10-50x faster than CPU.\n");
    printf("   Current speedup: %.1fx\n", cpu_matmul_time / gpu_matmul_time);
    
    if (cpu_matmul_time / gpu_matmul_time < 2.0) {
        printf("\n   ⚠️  WARNING: GPU speedup is low (< 2x)\n");
        printf("   This suggests GPU may not be fully utilized.\n");
        printf("   Possible reasons:\n");
        printf("   - Small matrices have high overhead\n");
        printf("   - Pipeline creation overhead per operation\n");
        printf("   - Memory transfer overhead\n");
        printf("   - GPU not actually being used\n");
    } else {
        printf("\n   ✓ GPU appears to be working (%.1fx speedup)\n", 
               cpu_matmul_time / gpu_matmul_time);
    }
    
    // Test with larger matrices
    printf("\n6. Large Matrix Test (512x512 @ 512x512):\n");
    Tensor *large_A = tensor_randn((size_t[]){512, 512}, 2, 47);
    Tensor *large_B = tensor_randn((size_t[]){512, 512}, 2, 48);
    
    // Warmup
    Tensor *large_warmup = tensor_matmul(large_A, large_B);
    tensor_free(large_warmup);
    
    start = get_time_ms();
    Tensor *large_gpu = tensor_matmul(large_A, large_B);
    double large_gpu_time = get_time_ms() - start;
    
    start = get_time_ms();
    Tensor *large_cpu = cpu_matmul(large_A, large_B);
    double large_cpu_time = get_time_ms() - start;
    
    printf("   GPU: %.2f ms\n", large_gpu_time);
    printf("   CPU: %.2f ms\n", large_cpu_time);
    printf("   Speedup: %.1fx\n", large_cpu_time / large_gpu_time);
    
    if (large_cpu_time / large_gpu_time < 10.0) {
        printf("\n   ⚠️  WARNING: Large matrix speedup is low (< 10x)\n");
        printf("   GPU may not be executing - check implementation!\n");
    } else {
        printf("   ✓ GPU showing good performance on large matrices\n");
    }
    
    // Cleanup
    tensor_free(input);
    tensor_free(W1);
    tensor_free(W2);
    tensor_free(relu_input);
    tensor_free(batch);
    tensor_free(h1);
    tensor_free(large_A);
    tensor_free(large_B);
    tensor_free(large_gpu);
    tensor_free(large_cpu);
    network_free(net);
    
    basednn_cleanup();
    
    return 0;
}
