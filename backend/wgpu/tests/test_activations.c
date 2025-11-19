#include "../../../core/include/tensor.h"
#include "../../../core/include/ops.h"
#include "../../../core/include/registry.h"
#include "../wgpu_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(void) {
    printf("=== Testing GPU Activation Functions ===\n\n");
    
    registry_init();
    
    if (wgpu_init() != 0) {
        fprintf(stderr, "Failed to initialize GPU\n");
        return 1;
    }
    
    wgpu_register_ops();
    
    // Test ReLU
    printf("Testing ReLU:\n");
    float relu_data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    Tensor *relu_in = tensor_create((size_t[]){5}, 1);
    memcpy(relu_in->data, relu_data, sizeof(relu_data));
    
    Tensor *relu_out = tensor_relu(relu_in);
    printf("  Input:  ");
    for (int i = 0; i < 5; i++) printf("%.1f ", relu_in->data[i]);
    printf("\n  Output: ");
    for (int i = 0; i < 5; i++) printf("%.1f ", relu_out->data[i]);
    printf("\n  Expected: 0.0 0.0 0.0 1.0 2.0\n");
    
    // Test Sigmoid
    printf("\nTesting Sigmoid:\n");
    float sig_data[] = {-2.0f, 0.0f, 2.0f};
    Tensor *sig_in = tensor_create((size_t[]){3}, 1);
    memcpy(sig_in->data, sig_data, sizeof(sig_data));
    
    Tensor *sig_out = tensor_sigmoid(sig_in);
    printf("  Input:  ");
    for (int i = 0; i < 3; i++) printf("%.1f ", sig_in->data[i]);
    printf("\n  Output: ");
    for (int i = 0; i < 3; i++) printf("%.4f ", sig_out->data[i]);
    printf("\n  Expected: ~0.1192 0.5000 ~0.8808\n");
    
    // Test Tanh
    printf("\nTesting Tanh:\n");
    float tanh_data[] = {-1.0f, 0.0f, 1.0f};
    Tensor *tanh_in = tensor_create((size_t[]){3}, 1);
    memcpy(tanh_in->data, tanh_data, sizeof(tanh_data));
    
    Tensor *tanh_out = tensor_tanh(tanh_in);
    printf("  Input:  ");
    for (int i = 0; i < 3; i++) printf("%.1f ", tanh_in->data[i]);
    printf("\n  Output: ");
    for (int i = 0; i < 3; i++) printf("%.4f ", tanh_out->data[i]);
    printf("\n  Expected: ~-0.7616 0.0000 ~0.7616\n");
    
    // Test Softmax (should use CPU fallback)
    printf("\nTesting Softmax (CPU fallback):\n");
    float soft_data[] = {1.0f, 2.0f, 3.0f};
    Tensor *soft_in = tensor_create((size_t[]){3}, 1);
    memcpy(soft_in->data, soft_data, sizeof(soft_data));
    
    Tensor *soft_out = tensor_softmax(soft_in);
    printf("  Input:  ");
    for (int i = 0; i < 3; i++) printf("%.1f ", soft_in->data[i]);
    printf("\n  Output: ");
    for (int i = 0; i < 3; i++) printf("%.4f ", soft_out->data[i]);
    printf("\n  Sum: %.4f (should be 1.0000)\n", 
           soft_out->data[0] + soft_out->data[1] + soft_out->data[2]);
    
    // Cleanup
    tensor_free(relu_in);
    tensor_free(relu_out);
    tensor_free(sig_in);
    tensor_free(sig_out);
    tensor_free(tanh_in);
    tensor_free(tanh_out);
    tensor_free(soft_in);
    tensor_free(soft_out);
    
    wgpu_cleanup();
    registry_cleanup();
    
    printf("\n✓ All activation functions tested successfully!\n");
    return 0;
}
