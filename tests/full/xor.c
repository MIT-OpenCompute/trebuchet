#include "lessbasednn/lessbasednn.h"
#include <stdio.h>

// ai generated xor test

int main() {
    Network *net = network_create();
    network_add_layer(net, layer_create(LINEAR(2, 4)));
    network_add_layer(net, layer_create(TANH()));
    network_add_layer(net, layer_create(LINEAR(4, 1)));
    network_add_layer(net, layer_create(SIGMOID()));
    
    Optimizer *opt = optimizer_create(net, ADAM(0.1f, 0.9f, 0.999f, 1e-8f));
    
    size_t input_shape[] = {4, 2};
    Tensor *inputs = tensor_create(input_shape, 2);
    inputs->data[0] = 0.0f; inputs->data[1] = 0.0f;
    inputs->data[2] = 0.0f; inputs->data[3] = 1.0f;
    inputs->data[4] = 1.0f; inputs->data[5] = 0.0f;
    inputs->data[6] = 1.0f; inputs->data[7] = 1.0f;
    
    size_t target_shape[] = {4, 1};
    Tensor *targets = tensor_create(target_shape, 2);
    targets->data[0] = 0.0f;
    targets->data[1] = 1.0f;
    targets->data[2] = 1.0f;
    targets->data[3] = 0.0f;
    
    network_train(net, opt, inputs, targets, 1000, 4, LOSS_MSE, 1);
    
    Tensor *predictions = network_forward(net, inputs);
    float accuracy = network_accuracy(predictions, targets);
    printf("\nAccuracy: %.2f%%\n", accuracy * 100.0f);
    
    tensor_free(inputs);
    tensor_free(targets);
    tensor_free(predictions);
    optimizer_free(opt);
    network_free(net);
    return 0;
}
