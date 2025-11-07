#ifndef NETWORK_H
#define NETWORK_H

#include "tensor.h"
#include "layer.h"

typedef struct Network Network;

struct Network {
    Layer **layers;
    size_t num_layers;
    size_t capacity;
}; 

// Network management
Network* network_create();
void network_add_layer(Network *net, Layer *layer);
void network_free(Network *net);

// Forward pass 
Tensor* network_forward(Network *net, Tensor *input);

// Utilities
void network_zero_grad(Network *net);
void network_print(Network *net);
Tensor** network_get_parameters(Network *net, size_t *num_params);

// Save/load network
void network_save(Network *net, const char *file_path);
Network* network_load(const char *file_path);

#endif