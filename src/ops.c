#include "basednn/ops.h"
#include "basednn/autograd.h"
#include <stdlib.h>
#include <math.h>

// Helper functions for gradient updates
static void grad_update_three_vars(Tensor *W, Tensor *X, Tensor *b, Tensor *Z, float (*func)(float, float), OpType op_type, void (*backward_fn)(Tensor *)) {
    if (W->requires_grad || X->requires_grad || b->requires_grad) {
        Z->requires_grad = 1;
        Z->op = op_type;
        Z->num_inputs = 3;
        Z->inputs = (Tensor **)malloc(3 * sizeof(Tensor *));
        Z->inputs[0] = W;
        Z->inputs[1] = X;
        Z->inputs[2] = b;
        Z->backward_fn = backward_fn;
    }
}

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


// Tensor functions
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
}

static float add_func(float x, float y) { return x + y; }
static float sub_func(float x, float y) { return x - y; }
static float mul_func(float x, float y) { return x * y; }

Tensor* tensor_add(Tensor *A, Tensor *B) {
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    // temporary solution, should probably add broadcasting support for other operations too
    if (A->ndim == 2 && B->ndim == 1 && A->shape[1] == B->shape[0]) {
        Tensor *C = tensor_create(A->shape, A->ndim);
        if (!C) return NULL;
        
        for (size_t i = 0; i < A->shape[0]; i++) {
            for (size_t j = 0; j < A->shape[1]; j++) {
                C->data[i * A->shape[1] + j] = A->data[i * A->shape[1] + j] + B->data[j];
            }
        }
        
        grad_update_two_vars(A, B, C, add_func, OP_ADD, backward_add);
        return C;
    }

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
    
    if (A->ndim == 1 && B->ndim == 1) {
        if (A->shape[0] != B->shape[0]) return NULL;
        
        Tensor *C = tensor_create((size_t[]){1}, 1);
        if (!C) return NULL;
        
        float acc = 0.0f;
        for (size_t i = 0; i < A->shape[0]; i++) {
            acc += A->data[i] * B->data[i];
        }
        C->data[0] = acc;
        
        grad_update_two_vars(A, B, C, NULL, OP_MATMUL, backward_matmul);
        return C;
    } else if (A->ndim == 2 && B->ndim == 1) {
        if (A->shape[1] != B->shape[0]) return NULL;
        
        Tensor *C = tensor_create((size_t[]){A->shape[0]}, 1);
        if (!C) return NULL;
        
        for (size_t i = 0; i < A->shape[0]; i++) {
            float acc = 0.0f;
            for (size_t k = 0; k < A->shape[1]; k++) {
                acc += A->data[i * A->shape[1] + k] * B->data[k];
            }
            C->data[i] = acc;
        }
        
        grad_update_two_vars(A, B, C, NULL, OP_MATMUL, backward_matmul);
        return C;
    } else if (A->ndim == 1 && B->ndim == 2) {
        if (A->shape[0] != B->shape[0]) return NULL;
        
        Tensor *C = tensor_create((size_t[]){B->shape[1]}, 1);
        if (!C) return NULL;
        
        for (size_t j = 0; j < B->shape[1]; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < A->shape[0]; k++) {
                acc += A->data[k] * B->data[k * B->shape[1] + j];
            }
            C->data[j] = acc;
        }
        
        grad_update_two_vars(A, B, C, NULL, OP_MATMUL, backward_matmul);
        return C;
    } else if (A->ndim == 2 && B->ndim == 2) {
        if (A->shape[1] != B->shape[0]) return NULL;

        size_t C_shape[2] = {A->shape[0], B->shape[1]}; 
        Tensor *C = tensor_create(C_shape, 2);
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
    } else {
        return NULL; 
    }
}

Tensor* tensor_transpose(Tensor *A) {
    if (!A) return NULL; 
    if (A->ndim != 2) return NULL; 

    size_t C_shape[2] = {A->shape[1], A->shape[0]}; 
    Tensor *C = tensor_create(C_shape, 2);
    if (!C) return NULL;

    for (size_t i = 0; i < A->shape[0]; i++) {
        for (size_t j = 0; j < A->shape[1]; j++) {
            C->data[j * C->shape[1] + i] = A->data[i * A->shape[1] + j]; 
        }
    }

    grad_update_one_var(A, C, NULL, OP_TRANSPOSE, backward_transpose);

    return C; 
}

// Activation functions
Tensor* tensor_relu(Tensor *Z) {
    if (!Z) return NULL;

    Tensor *A = tensor_create(Z->shape, Z->ndim); 
    if (!A) return NULL; 

    for (size_t i = 0; i < Z->size; i++) {
        A->data[i] = Z->data[i] > 0.0f ? Z->data[i] : 0.0f; 
    }

    grad_update_one_var(Z, A, NULL, OP_RELU, backward_relu);

    return A; 
}

Tensor* tensor_sigmoid(Tensor *Z) {
    if (!Z) return NULL; 

    Tensor *A = tensor_create(Z->shape, Z->ndim);
    if (!A) return NULL;

    for (size_t i = 0; i < Z->size; i++) {
        A->data[i] = 1.0f / (1.0f + expf(-Z->data[i])); 
    }

    grad_update_one_var(Z, A, NULL, OP_SIGMOID, backward_sigmoid);

    return A;
}

Tensor* tensor_tanh(Tensor *Z) {
    if (!Z) return NULL; 

    Tensor *A = tensor_create(Z->shape, Z->ndim);
    if (!A) return NULL;

    for (size_t i = 0; i < Z->size; i++) {
        A->data[i] = tanhf(Z->data[i]); 
    }

    grad_update_one_var(Z, A, NULL, OP_TANH, backward_tanh);

    return A;
}

Tensor* tensor_softmax(Tensor *Z) {
    if (!Z) return NULL; 
    
    Tensor *A = tensor_create(Z->shape, Z->ndim);
    if (!A) return NULL; 

    float max_val = Z->data[0]; 
    for (size_t i = 1; i < Z->size; i++) {
        if (Z->data[i] > max_val) max_val = Z->data[i];
    }

    float sum = 0.0f; 
    for (size_t i = 0; i < Z->size; i++) {
        A->data[i] = expf(Z->data[i] - max_val);
        sum += A->data[i];
    }

    for (size_t i = 0; i < Z->size; i++) {
        A->data[i] /= sum; 
    }

    grad_update_one_var(Z, A, NULL, OP_SOFTMAX, backward_softmax);

    return A;
}

// Loss functions
static int check_pred_target(Tensor *predictions, Tensor *targets) {
    if (!predictions || !targets) return 0; 
    if (predictions->ndim != targets->ndim) return 0; 
    for (size_t i = 0; i < predictions->ndim; i++) {
        if (predictions->shape[i] != targets->shape[i]) return 0; 
    }
    return 1; 
}

Tensor* tensor_mse(Tensor *predictions, Tensor *targets) {
    if (!check_pred_target(predictions, targets)) return NULL;

    Tensor *loss = tensor_create((size_t[]){1}, 1);
    if (!loss) return NULL; 

    float sum_sq_error = 0.0f; 
    for (size_t i = 0; i < predictions->size; i++) {
        float diff = predictions->data[i] - targets->data[i];
        sum_sq_error += diff * diff;
    }
    loss->data[0] = sum_sq_error / predictions->size;
    return loss;
}

Tensor* tensor_cross_entropy(Tensor *predictions, Tensor *targets) {
    if (!check_pred_target(predictions, targets)) return NULL; 
    
    Tensor *loss = tensor_create((size_t[]){1}, 1);
    if (!loss) return NULL;

    float sum_ce_loss = 0.0f; 
    for (size_t i = 0; i < predictions->size; i++) {
        float pred = predictions->data[i];
        float target = targets->data[i]; 
        sum_ce_loss += -target * logf(pred > 0.0f ? pred : 0.0f);
    }
    loss->data[0] = sum_ce_loss / predictions->size; 
    return loss;
}

Tensor* tensor_binary_cross_entropy(Tensor *predictions, Tensor *targets) {
    if (!check_pred_target(predictions, targets)) return NULL; 
    
    Tensor *loss = tensor_create((size_t[]){1}, 1);
    if (!loss) return NULL;

    float sum_bce_loss = 0.0f; 
    for (size_t i = 0; i < predictions->size; i++) {
        float pred = predictions->data[i];
        float target = targets->data[i]; 
        sum_bce_loss += - (target * logf(pred > 0.0f ? pred : 0.0f) + (1.0f - target) * logf((1.0f - pred) > 0.0f ? (1.0f - pred) : 0.0f));
    }
    loss->data[0] = sum_bce_loss / predictions->size; 
    return loss;
}