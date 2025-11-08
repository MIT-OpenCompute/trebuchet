#include "lessbasednn/tensor.h"
#include "lessbasednn/autograd.h"
#include <stdlib.h> 
#include <stdio.h>
#include <string.h>
#include <math.h>


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
    T->owns_data = 1;
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

    if (T->owns_data) {
        if (T->data) free(T->data); 
        if (T->grad) free(T->grad); 
    }

    if (T->shape) free(T->shape); 
    if (T->inputs) free(T->inputs); 
    if (T->extra_data) free(T->extra_data);

    free(T); 
}

// Autograd functions
void tensor_set_requires_grad(Tensor *T, int requires_grad) {
    if (!T) return; 
    T->requires_grad = requires_grad;
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

void tensor_zero_grad(Tensor *T) {
    if (!T || !T->grad) return;
    memset(T->grad, 0, T->size * sizeof(float));
}

void tensor_fill(Tensor *T, float value) {
    if (!T) return;
    for (size_t i = 0; i < T->size; i++) {
        T->data[i] = value;
    }
}

void tensor_print(Tensor *T) {
    if (!T) return;

    printf("Tensor(shape=[");
    for (size_t i = 0; i < T->ndim; i++) {
        printf("%zu", T->shape[i]);
        if (i < T->ndim - 1) {
            printf(", ");
        }
    }
    printf("], ndim=%zu, data=\n[", T->ndim);
    if (T->ndim == 2) {
        for (size_t i = 0; i < T->shape[0]; i++) {
            printf("[");
            for (size_t j = 0; j < T->shape[1]; j++) {
                printf("%.4f", T->data[i * T->shape[1] + j]);
                if (j < T->shape[1] - 1) {
                    printf(", ");
                }
            }
            printf("]");
            if (i < T->shape[0] - 1) {
                printf(",\n ");
            }
        }
    } else {
        for (size_t i = 0; i < T->size; i++) {
            printf("%.4f", T->data[i]);
            if (i < T->size - 1) {
                printf(", ");
            }
        }
    }
    printf("])\n");
}

Tensor* tensor_copy(Tensor *T) {
    if (!T) return NULL;

    Tensor *C = tensor_create(T->shape, T->ndim);
    if (!C) return NULL;

    memcpy(C->data, T->data, T->size * sizeof(float));

    C->grad = NULL; 
    C->requires_grad = 0;
    C->owns_data = 1; 
    C->op = OP_NONE;
    C->inputs = NULL;
    C->num_inputs = 0;
    C->backward_fn = NULL;
    C->extra_data = NULL;

    return C;
}