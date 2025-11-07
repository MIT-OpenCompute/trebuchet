#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

typedef struct Tensor Tensor;

typedef enum {
    OP_NONE,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_MATMUL,
    OP_TRANSPOSE,
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_SOFTMAX,
    OP_MSE,
    OP_CROSS_ENTROPY,
    OP_BINARY_CROSS_ENTROPY
} OpType;

struct Tensor {
    float *data;
    float *grad;
    size_t *shape;
    size_t ndim;
    size_t size;
    
    int requires_grad;
    OpType op;
    Tensor **inputs;
    size_t num_inputs;
    void (*backward_fn)(Tensor *self);
    void *extra_data;
};

// Tensor creation/destruction
Tensor* tensor_create(size_t *shape, size_t ndim);
Tensor* tensor_zeroes(size_t *shape, size_t ndim);
Tensor* tensor_ones(size_t *shape, size_t ndim);
Tensor* tensor_randn(size_t *shape, size_t ndim, int seed);
void tensor_free(Tensor *T);

// Autograd operations
void tensor_set_requires_grad(Tensor *T, int requires_grad);
void tensor_zero_grad(Tensor *T);
void tensor_backward(Tensor *T);

// Utilities
void tensor_print(Tensor *T);
Tensor* tensor_copy(Tensor *T); 

#endif