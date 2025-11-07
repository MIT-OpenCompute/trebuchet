#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"

typedef enum {
    LAYER_LINEAR,
    LAYER_CONV2D,
    LAYER_MAXPOOL2D, 
    LAYER_FLATTEN
} LayerType; 

typedef struct {
    LayerType type; 
    Tensor *weights;
    Tensor *bias;
    Tensor *output; 

    void (*forward)(struct Layer *self, Tensor *input); 
    void (*backward)(struct Layer *self, Tensor *grad_output);
    void (*free)(struct Layer *self); 
} Layer; 

typedef struct {
    Layer base; 
    size_t in_features; 
    size_t out_features; 
    Tensor *input_cache; 
} LinearLayer; 

// Linear layer constructors/destructors
LinearLayer* layer_linear_create(size_t in_features, size_t out_features);
void layer_linear_free(LinearLayer *self);

// Linear layer operations
void layer_linear_forward(Layer *self, Tensor *input);
void layer_linear_backward(Layer *self, Tensor *grad_output);

// Other layer type operations
void layer_forward(Layer *layer, Tensor *input); 
void layer_backward(Layer *layer, Tensor *grad_output); 
void layer_free(Layer *layer); 

#endif