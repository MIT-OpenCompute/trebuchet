#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

// Tensor function gradients
void backward_add(Tensor *C);
void backward_sub(Tensor *C);
void backward_mul(Tensor *C);
void backward_matmul(Tensor *C);
void backward_transpose(Tensor *C);

// Activation function gradients
void backward_relu(Tensor *A);
void backward_sigmoid(Tensor *A);
void backward_tanh(Tensor *A);
void backward_softmax(Tensor *A);

// Loss function gradients
void backward_mse(Tensor *L);
void backward_cross_entropy(Tensor *L);
void backward_binary_cross_entropy(Tensor *L);

#endif
