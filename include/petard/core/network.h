#ifndef NETWORK_H
#define NETWORK_H

#include "tensor.h"
#include "layer.h"
#include "conv.h"

typedef struct {
    Layer **layers; 
    size_t num_layers; 
    size_t capacity; 
} Network;

// Network management
Network* network_create();
void network_add_layer(Network *net, Layer *layer); 
void network_free(Network *net); 

// Forward/backward operations
Tensor* network_forward(Network *net, Tensor *input); 
void network_backward(Network *net, Tensor *grad_output); 

// Training utilities
void network_zero_grad(Network *net); 
void network_print(Network *net);

// Save/load network
void network_save(Network *net, const char *filepath); 
Network* network_load(const char *filepath); 

#endif