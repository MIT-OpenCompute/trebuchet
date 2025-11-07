#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include "ops.h"

typedef enum {
    LAYER_LINEAR,
    LAYER_RELU,
    LAYER_SIGMOID,
    LAYER_TANH,
    LAYER_SOFTMAX, 
} LayerType; 

typedef struct {
    LayerType type; 
    union {
        struct {
            size_t in_features; 
            size_t out_features; 
        } linear;
    } params; 
} LayerConfig;

#define LINEAR(in_features, out_features) (LayerConfig){ .type = LAYER_LINEAR, .params.linear = { in_features, out_features } }
#define RELU() (LayerConfig){ .type = LAYER_RELU }
#define SIGMOID() (LayerConfig){ .type = LAYER_SIGMOID }
#define TANH() (LayerConfig){ .type = LAYER_TANH }
#define SOFTMAX() (LayerConfig){ .type = LAYER_SOFTMAX }

typedef struct Layer Layer;

struct Layer {
    LayerType type;
    Tensor *weights;
    Tensor *bias;
    Tensor *output;
    Tensor **parameters;
    size_t num_parameters;
    Tensor* (*forward)(Layer *self, Tensor *input);
    void (*free)(Layer *self);
}; 

// Layer constructors/destructor
Layer* layer_create(LayerConfig config);
void layer_free(Layer *layer); 

// Layer operations
Tensor* layer_forward(Layer *layer, Tensor *input);

// Utilities
void layer_zero_grad(Layer *layer);
Tensor** layer_get_parameters(Layer *layer, size_t *num_params);

#endif