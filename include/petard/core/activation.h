#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "tensor.h"
#include "layer.h"

typedef struct {
    Layer base; 
    Tensor *input_cache; 
    void (*activate)(Tensor *input, Tensor *output); 
    void (*derivative)(Tensor *input, Tensor *grad_output, Tensor *grad_input); 
} ActivationLayer;

// Activation functions
void activation_relu(Tensor *input, Tensor *output);
void activation_sigmoid(Tensor *input, Tensor *output);
void activation_tanh(Tensor *input, Tensor *output);
void activation_softmax(Tensor *input, Tensor *output);

// Gradients 
void activation_relu_backward(Tensor *input, Tensor *grad_output, Tensor *grad_input); 
void activation_sigmoid_backward(Tensor *input, Tensor *grad_output, Tensor *grad_input);
void activation_tanh_backward(Tensor *input, Tensor *grad_output, Tensor *grad_input);
void activation_softmax_backward(Tensor *input, Tensor *grad_output, Tensor *grad_input);

// Layer constructors
ActivationLayer* activation_relu_create(); 
ActivationLayer* activation_sigmoid_create();
ActivationLayer* activation_tanh_create();
ActivationLayer* activation_softmax_create();

#endif