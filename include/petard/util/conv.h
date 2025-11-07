#ifndef PETARD_CONV_H
#define PETARD_CONV_H

#include "layer.h"

typedef struct {
    Layer base; 
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    Tensor *input_cache;
} Conv2DLayer;

typedef struct {
    Layer base; 
    size_t pool_size; 
    size_t stride; 
    Tensor *input_cache; 
} MaxPool2DLayer;

typedef struct {
    Layer base; 
    size_t input_size; 
} FlattenLayer;


// Conv2D layer constructors/destructors
Conv2DLayer* layer_conv2d_create(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride, size_t padding);
void layer_conv2d_free(Conv2DLayer *self);

// Conv2D layer operations
void layer_conv2d_forward(Layer *self, Tensor *input);
void layer_conv2d_backward(Layer *self, Tensor *grad_output);

// MaxPool2D layer constructors/destructors
MaxPool2DLayer* layer_maxpool2d_create(size_t pool_size, size_t stride);
void layer_maxpool2d_free(MaxPool2DLayer *self);

// MaxPool2D layer operations
void layer_maxpool2d_forward(Layer *self, Tensor *input);
void layer_maxpool2d_backward(Layer *self, Tensor *grad_output);

// Flatten layer constructors/destructors
FlattenLayer* layer_flatten_create(size_t input_size);
void layer_flatten_free(FlattenLayer *self);

// Flatten layer operations
void layer_flatten_forward(Layer *self, Tensor *input);
void layer_flatten_backward(Layer *self, Tensor *grad_output);

#endif