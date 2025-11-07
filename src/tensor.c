#include "tensor.h"
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include <math.h> 

// Autograd forward declarations of backwards
static void backward_add(Tensor *out); 
static void backward_sub(Tensor *out);
static void backward_mul(Tensor *out);
static void backward_matmul(Tensor *out);
static void backward_transpose(Tensor *out);
static void backward_relu(Tensor *out); 
static void backward_sigmoid(Tensor *out); 
static void backward_tanh(Tensor *out); 
static void backward_softmax(Tensor *out);


// Autograd helper functions
static void topological_sort_util(Tensor *T, Tensor **visited, size_t *visited_count, Tensor **stack, size_t *stack_count, size_t max_size) {
    for (size_t i = 0; i < *visited_count; i++) {
        if (visited[i] == T) return; 
    }

    visited[(*visited_count)++] = T; 

    if (T->inputs) {
        for (size_t i = 0; i < T->num_inputs; i++) {
            if (T->inputs[i] && T->inputs[i]->requires_grad) {
                topological_sort_util(T->inputs[i], visited, visited_count, stack, stack_count, max_size); 
            }
        }
    }

    if (*stack_count < max_size) {
        stack[(*stack_count)++] = T;
    }
}

// Tensor creation/destruction
Tensor* tensor_create(size_t *shape, size_t ndim) {
    Tensor *T = (Tensor *)malloc(sizeof(Tensor)); 
    if (!T) return NULL; 

    T->ndim = ndim; 
    T->shape = (size_t *)malloc(ndim * sizeof(size_t)); 
    if (!T->shape) {
        free(T);
        return NULL;
    }

    T->size = 1; 
    for (size_t i = 0; i < ndim; i++) {
        T->shape[i] = shape[i]; 
        T->size *= shape[i]; 
    }

    T->data = (float *)malloc(T->size * sizeof(float)); 
    if (!T->data) {
        free(T->shape);
        free(T);
        return NULL;
    }

    T->grad = NULL; 
    T->requires_grad = 0;
    T->op = OP_NONE;
    T->inputs = NULL;
    T->num_inputs = 0;
    T->backward_fn = NULL;
    T->extra_data = NULL;
    return T; 
}

Tensor* tensor_zeroes(size_t *shape, size_t ndim) {
    Tensor *T = tensor_create(shape, ndim); 
    if (!T) return NULL; 

    memset(T->data, 0, T->size * sizeof(float)); 
    return T; 
}

Tensor* tensor_ones(size_t *shape, size_t ndim) {
    Tensor *T = tensor_create(shape, ndim); 
    if (!T) return NULL; 

    for (size_t i = 0; i < T->size; i++) {
        T->data[i] = 1.0f; 
    }
    return T;
}

Tensor* tensor_randn(size_t *shape, size_t ndim, int seed) {
    Tensor *T = tensor_create(shape, ndim); 
    if (!T) return NULL; 

    srand(seed); 
    for (size_t i = 0; i < T->size; i++) {
        float u1 = ((float)rand() / RAND_MAX);
        float u2 = ((float)rand() / RAND_MAX);
        T->data[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2); 
    }
    return T; 
}

void tensor_free(Tensor *T) {
    if (!T) return; 
    if (T->data) free(T->data); 
    if (T->grad) free(T->grad); 
    if (T->shape) free(T->shape); 
    if (T->inputs) free(T->inputs); 
    free(T); 
}

// Autograd functions
void tensor_set_requires_grad(Tensor *T, int requires_grad) {
    if (!T) return; 
    if (T->grad) {
        memset(T->grad, 0, T->size * sizeof(float)); 
    }
}

void tensor_backward(Tensor *T) {
    if (!T || !T->requires_grad) return; 

    if (!T->grad) {
        T->grad = (float *)malloc(T->size * sizeof(float)); 
        for (size_t i = 0; i < T->size; i++) {
            T->grad[i] = 1.0f; 
        }
    }

    size_t max_size = 1000; 
    Tensor **visited = (Tensor **)malloc(max_size * sizeof(Tensor *)); 
    Tensor **stack = (Tensor **)malloc(max_size * sizeof(Tensor *));
    size_t visited_count = 0;
    size_t stack_count = 0;

    topological_sort_util(T, visited, &visited_count, stack, &stack_count, max_size); 

    for (size_t i = stack_count; i > 0; i--) {
        Tensor *node = stack[i - 1]; 
        if (node->backward_fn) {
            node->backward_fn(node);
        }
    }

    free(visited); 
    free(stack); 
}


// Tensor operations
static void grad_update_two_vars(Tensor *A, Tensor *B, Tensor *C, float (*func)(float, float), OpType op_type, void (*backward_fn)(Tensor *)) {
    if (A->requires_grad || B->requires_grad) {
        C->requires_grad = 1;
        C->op = op_type;
        C->num_inputs = 2;
        C->inputs = (Tensor **)malloc(2 * sizeof(Tensor *));
        C->inputs[0] = A;
        C->inputs[1] = B;
        C->backward_fn = backward_fn;
    }
}

static void grad_update_one_var(Tensor *A, Tensor *C, float (*func)(float, float), OpType op_type, void (*backward_fn)(Tensor *)) {
    if (A->requires_grad) {
        C->requires_grad = 1;
        C->op = op_type;
        C->num_inputs = 1;
        C->inputs = (Tensor **)malloc(1 * sizeof(Tensor *));
        C->inputs[0] = A;
        C->backward_fn = backward_fn;
    }
}

static void tensor_ewise(Tensor *A, Tensor *B, Tensor *C, float (*func)(float, float), OpType op_type, void (*backward_fn)(Tensor *)) {
    if (!A || !B || !C) return; 
    if (A->ndim != B->ndim || A->ndim != C->ndim) return; 
    for (size_t i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != C->shape[i]) return;
    }

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = func(A->data[i], B->data[i]); 
    }

    grad_update_two_vars(A, B, C, func, op_type, backward_fn);

    return C; 
}

static float add_func(float x, float y) { return x + y; }
static float sub_func(float x, float y) { return x - y; }
static float mul_func(float x, float y) { return x * y; }

Tensor* tensor_add(Tensor *A, Tensor *B) {
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    tensor_ewise(A, B, C, add_func, OP_ADD, backward_add);
    return C;
}

Tensor* tensor_sub(Tensor *A, Tensor *B) {
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    tensor_ewise(A, B, C, sub_func, OP_SUB, backward_sub);
    return C;
}

Tensor* tensor_mul(Tensor *A, Tensor *B) {
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    tensor_ewise(A, B, C, mul_func, OP_MUL, backward_mul);
    return C;
}

Tensor* tensor_matmul(Tensor *A, Tensor *B) {
    if (!A || !B) return NULL; 
    if (A->ndim != 2 || B->ndim != 2) return NULL; 
    if (A->shape[1] != B->shape[0]) return NULL; 

    size_t out_shape[2] = {A->shape[0], B->shape[1]}; 
    Tensor *C = tensor_create(out_shape, 2);
    if (!C) return NULL; 

    for (size_t i = 0; i < A->shape[0]; i++) {
        for (size_t j = 0; j < B->shape[1]; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < A->shape[1]; k++) {
                acc += A->data[i * A->shape[1] + k] * B->data[k * B->shape[1] + j];
            }
            C->data[i * C->shape[1] + j] = acc; 
        }
    }

    grad_update_two_vars(A, B, C, NULL, OP_MATMUL, backward_matmul);

    return C; 
}

Tensor* tensor_transpose(Tensor *A) {
    if (!A) return NULL; 
    if (A->ndim != 2) return NULL; 

    size_t out_shape[2] = {A->shape[1], A->shape[0]}; 
    Tensor *C = tensor_create(out_shape, 2);
    if (!C) return NULL;

    for (size_t i = 0; i < A->shape[0]; i++) {
        for (size_t j = 0; j < A->shape[1]; j++) {
            C->data[j * C->shape[1] + i] = A->data[i * A->shape[1] + j]; 
        }
    }

    grad_update_one_var(A, C, NULL, OP_TRANSPOSE, backward_transpose);

    return C; 
}

// Activation operations
Tensor* tensor_relu(Tensor *A) {
    if (!A) return NULL;

    Tensor *C = tensor_create(A->shape, A->ndim); 
    if (!C) return NULL; 

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = A->data[i] > 0.0f ? A->data[i] : 0.0f; 
    }

    grad_update_one_var(A, C, NULL, OP_RELU, backward_relu);

    return C; 
}

Tensor* tensor_sigmoid(Tensor *A) {
    if (!A) return NULL; 

    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = 1.0f / (1.0f + expf(-A->data[i])); 
    }

    grad_update_one_var(A, C, NULL, OP_SIGMOID, backward_sigmoid);

    return C;
}

Tensor* tensor_tanh(Tensor *A) {
    if (!A) return NULL; 

    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = tanhf(A->data[i]); 
    }

    grad_update_one_var(A, C, NULL, OP_TANH, backward_tanh);

    return C;
}

Tensor* tensor_softmax(Tensor *A) {
    if (!A) return NULL; 
    
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL; 

    float max_val = A->data[0]; 
    for (size_t i = 1; i < A->size; i++) {
        if (A->data[i] > max_val) max_val = A->data[i];
    }

    float sum = 0.0f; 
    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = expf(A->data[i] - max_val);
        sum += C->data[i];
    }

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] /= sum; 
    }

    grad_update_one_var(A, C, NULL, OP_SOFTMAX, backward_softmax);

    return C;
}

// Utilities
static void backward_add(Tensor *out) {
    Tensor *A = out->inputs[0];
    Tensor *B = out->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += out->grad[i];
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
        for (size_t i = 0; i < B->size; i++) {
            B->grad[i] += out->grad[i];
        }
    }
}

static void backward_sub(Tensor *out) {
    Tensor *A = out->inputs[0];
    Tensor *B = out->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += out->grad[i];
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
        for (size_t i = 0; i < B->size; i++) {
            B->grad[i] -= out->grad[i];
        }
    }
}

static void backward_mul(Tensor *out) {
    Tensor *A = out->inputs[0];
    Tensor *B = out->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += out->grad[i] * B->data[i];
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
        for (size_t i = 0; i < B->size; i++) {
            B->grad[i] += out->grad[i] * A->data[i];
        }
    }
}

static void backward_matmul(Tensor *out) {
    Tensor *A = out->inputs[0];
    Tensor *B = out->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->shape[0]; i++) {
            for (size_t k = 0; k < A->shape[1]; k++) {
                for (size_t j = 0; j < B->shape[1]; j++) {
                    A->grad[i * A->shape[1] + k] += 
                        out->grad[i * B->shape[1] + j] * B->data[k * B->shape[1] + j];
                }
            }
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
        for (size_t k = 0; k < B->shape[0]; k++) {
            for (size_t j = 0; j < B->shape[1]; j++) {
                for (size_t i = 0; i < A->shape[0]; i++) {
                    B->grad[k * B->shape[1] + j] += 
                        out->grad[i * B->shape[1] + j] * A->data[i * A->shape[1] + k];
                }
            }
        }
    }
}

static void backward_transpose(Tensor *out) {
    Tensor *A = out->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->shape[0]; i++) {
            for (size_t j = 0; j < A->shape[1]; j++) {
                A->grad[i * A->shape[1] + j] += out->grad[j * A->shape[0] + i];
            }
        }
    }
}

static void backward_relu(Tensor *out) {
    Tensor *A = out->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += out->grad[i] * (A->data[i] > 0 ? 1.0f : 0.0f);
        }
    }
}

static void backward_sigmoid(Tensor *out) {
    Tensor *A = out->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            float sig = out->data[i];
            A->grad[i] += out->grad[i] * sig * (1.0f - sig);
        }
    }
}

static void backward_tanh(Tensor *out) {
    Tensor *A = out->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            float t = out->data[i];
            A->grad[i] += out->grad[i] * (1.0f - t * t);
        }
    }
}

static void backward_softmax(Tensor *out) {
    Tensor *A = out->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            for (size_t j = 0; j < A->size; j++) {
                float delta = (i == j) ? 1.0f : 0.0f;
                A->grad[i] += out->grad[j] * out->data[j] * (delta - out->data[i]);
            }
        }
    }
}