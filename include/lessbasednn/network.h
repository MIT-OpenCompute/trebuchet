#ifndef NETWORK_H
#define NETWORK_H

#include "tensor.h"
#include "layer.h"
#include "optimizer.h"

typedef enum {
    LOSS_MSE,
    LOSS_CROSS_ENTROPY,
    LOSS_BINARY_CROSS_ENTROPY
} LossType;

struct Network {
    Layer **layers;
    Tensor **parameters;
    size_t num_layers;
    size_t num_parameters;
    size_t capacity;
}; 

// Network management
Network* network_create();
void network_add_layer(Network *net, Layer *layer);
void network_free(Network *net);

// Forward pass 
Tensor* network_forward(Network *net, Tensor *input);

// Training
void network_train(Network *net, Optimizer *opt, Tensor *inputs, Tensor *targets, size_t epochs, size_t batch_size, LossType loss_type, int verbose);
float network_train_step(Network *net, Tensor *input, Tensor *target, Optimizer *opt, LossType loss_type);
void network_zero_grad(Network *net);

// Utilities
void network_print(Network *net);
Tensor** network_get_parameters(Network *net, size_t *num_params);
float network_accuracy(Tensor *predictions, Tensor *targets);

// Save/load network
void network_save(Network *net, const char *file_path);
Network* network_load(const char *file_path);

#endif