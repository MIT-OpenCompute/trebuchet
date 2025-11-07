#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h> 

typedef struct {
    float *data; 
    float *grad; 
    size_t *shape; 
    size_t ndim; 
    size_t size; 
    int requires_grad;
} Tensor;

// Creation and destruction
Tensor* tensor_create(size_t *shape, size_t ndim); 
Tensor* tensor_zeroes(size_t *shape, size_t ndim); 
Tensor* tensor_ones(size_t *shape, size_t ndim); 
Tensor* tensor_randn(size_t *shape, size_t ndim, int seed); 
void tensor_free(Tensor *T); 

// Operations
Tensor* tensor_matmul(Tensor *A, Tensor *B);
Tensor* tensor_add(Tensor *A, Tensor *B); 
Tensor* tensor_sub(Tensor *A, Tensor *B);
Tensor* tensor_mul(Tensor *A, Tensor *B);
Tensor* tensor_transpose(Tensor *T); 

// Gradients
void tensor_zero_grad(Tensor *T); 
void tensor_backward(Tensor *T); 

// Utilities
void tensor_print(Tensor *T);
Tensor* tensor_copy(Tensor *T); 

#endif