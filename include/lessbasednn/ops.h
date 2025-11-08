#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// Tensor functions
Tensor* tensor_add(Tensor *A, Tensor *B);
Tensor* tensor_sub(Tensor *A, Tensor *B);
Tensor* tensor_mul(Tensor *A, Tensor *B);
Tensor* tensor_matmul(Tensor *A, Tensor *B);
Tensor* tensor_transpose(Tensor *A);

// Activation functions
Tensor* tensor_relu(Tensor *Z);
Tensor* tensor_sigmoid(Tensor *Z);
Tensor* tensor_tanh(Tensor *Z);
Tensor* tensor_softmax(Tensor *Z);

// Loss functions
Tensor* tensor_mse(Tensor *predictions, Tensor *targets);
Tensor* tensor_cross_entropy(Tensor *predictions, Tensor *targets);
Tensor* tensor_binary_cross_entropy(Tensor *predictions, Tensor *targets);

// Utilities
Tensor* tensor_slice(Tensor *input, size_t start, size_t end); 


#endif