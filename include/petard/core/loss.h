#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

typedef enum {
    LOSS_MSE, 
    LOSS_CROSS_ENTROPY, 
    LOSS_BINARY_CROSS_ENTROPY
} LossType;

typedef struct {
    LossType type; 
    float (*forward)(Tensor *predictions, Tensor *targets);
    void (*backward)(Tensor *predictions, Tensor *targets, Tensor *grad);
} Loss;

// Loss functions
float loss_mse(Tensor *predictions, Tensor *targets);
float loss_cross_entropy(Tensor *predictions, Tensor *targets); 
float loss_binary_cross_entropy(Tensor *predictions, Tensor *targets);

// Gradients
void loss_mse_backward(Tensor *predictions, Tensor *targets, Tensor *grad); 
void loss_cross_entropy_backward(Tensor *predictions, Tensor *targets, Tensor *grad); 
void loss_binary_cross_entropy_backward(Tensor *predictions, Tensor *targets, Tensor *grad); 

// Loss constructor/destructor
Loss* loss_create(LossType type); 
void loss_free(Loss *self);

// Loss operations
float loss_forward(Loss *self, Tensor *predictions, Tensor *targets);
void loss_backward(Loss *self, Tensor *predictions, Tensor *targets, Tensor *grad);

#endif