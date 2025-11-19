#include "../../include/basednn.h"
#include <stdio.h>
#include <sys/time.h>

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main() {
    printf("=== MNIST GPU Performance Test ===\n\n");
    basednn_init();
    
    // Create a simple network like MNIST
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(784, 256)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(256, 128)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(128, 10)));
    network_add_layer(net, layer_create(SOFTMAX()));
    
    // Create dummy batch of data (64 samples)
    Tensor *input = tensor_randn((size_t[]){64, 784}, 2, 42);
    
    printf("Running forward pass with batch size 64...\n");
    printf("Network: 784 -> 256 -> 128 -> 10\n\n");
    
    // Warmup
    Tensor *warmup = network_forward(net, input);
    tensor_free(warmup);
    
    // Time forward passes
    int num_iterations = 10;
    double start = get_time_ms();
    
    for (int i = 0; i < num_iterations; i++) {
        Tensor *output = network_forward(net, input);
        tensor_free(output);
    }
    
    double total_time = get_time_ms() - start;
    double avg_time = total_time / num_iterations;
    
    printf("Completed %d forward passes\n", num_iterations);
    printf("Total time: %.2f ms\n", total_time);
    printf("Average time per forward pass: %.2f ms\n", avg_time);
    printf("Throughput: %.1f samples/sec\n", (64.0 * num_iterations) / (total_time / 1000.0));
    
    tensor_free(input);
    network_free(net);
    basednn_cleanup();
    
    return 0;
}
