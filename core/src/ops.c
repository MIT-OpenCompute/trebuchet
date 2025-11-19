#include "../include/ops.h"
#include "../include/registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ====================================================
// Gradient Update Helpers
// ====================================================

static void grad_update_three_vars(Tensor *W, Tensor *X, Tensor *b, Tensor *Z, float (*func)(float, float), const char *op_name, void (*backward_fn)(Tensor *)) {
    if (W->requires_grad || X->requires_grad || b->requires_grad) {
        Z->requires_grad = 1;
        Z->op_name = op_name ? strdup(op_name) : NULL;
        Z->num_inputs = 3;
        Z->inputs = (Tensor **)malloc(3 * sizeof(Tensor *));
        Z->inputs[0] = W;
        Z->inputs[1] = X;
        Z->inputs[2] = b;
        Z->backward_fn = backward_fn;
    }
}

static void grad_update_two_vars(Tensor *A, Tensor *B, Tensor *C, float (*func)(float, float), const char *op_name, void (*backward_fn)(Tensor *)) {
    if (A->requires_grad || B->requires_grad) {
        C->requires_grad = 1;
        C->op_name = op_name ? strdup(op_name) : NULL;
        C->num_inputs = 2;
        C->inputs = (Tensor **)malloc(2 * sizeof(Tensor *));
        C->inputs[0] = A;
        C->inputs[1] = B;
        C->backward_fn = backward_fn;
    }
}

static void grad_update_one_var(Tensor *A, Tensor *C, float (*func)(float, float), const char *op_name, void (*backward_fn)(Tensor *)) {
    if (A->requires_grad) {
        C->requires_grad = 1;
        C->op_name = op_name ? strdup(op_name) : NULL;
        C->num_inputs = 1;
        C->inputs = (Tensor **)malloc(1 * sizeof(Tensor *));
        C->inputs[0] = A;
        C->backward_fn = backward_fn;
    }
}

// ====================================================
// Elementwise Operations
// ====================================================

static void tensor_ewise(Tensor *A, Tensor *B, Tensor *C, float (*func)(float, float), const char *op_name, void (*backward_fn)(Tensor *)) {
    if (!A || !B || !C) return; 
    if (A->ndim != B->ndim || A->ndim != C->ndim) return; 
    for (size_t i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i] || A->shape[i] != C->shape[i]) return;
    }

    for (size_t i = 0; i < A->size; i++) {
        C->data[i] = func(A->data[i], B->data[i]); 
    }

    grad_update_two_vars(A, B, C, func, op_name, backward_fn);
}

static float add_func(float x, float y) { return x + y; }
static float sub_func(float x, float y) { return x - y; }
static float mul_func(float x, float y) { return x * y; }

Tensor* tensor_add(Tensor *A, Tensor *B) {
    if (!A || !B) return NULL;
    
    // Handle broadcasting case (2D + 1D bias) on CPU only
    if (A->ndim == 2 && B->ndim == 1 && A->shape[1] == B->shape[0]) {
        Tensor *C = tensor_create(A->shape, A->ndim);
        if (!C) return NULL;
        
        for (size_t i = 0; i < A->shape[0]; i++) {
            for (size_t j = 0; j < A->shape[1]; j++) {
                C->data[i * A->shape[1] + j] = A->data[i * A->shape[1] + j] + B->data[j];
            }
        }
        
        grad_update_two_vars(A, B, C, NULL, "add", backward_add);
        return C;
    }
    
    // Check for backend implementation (e.g., GPU) for same-shape adds
    OpFn backend_fn = get_operation_fn("add");
    if (backend_fn) {
        return backend_fn(A, B);
    }
    
    // Fallback to CPU implementation
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    tensor_ewise(A, B, C, add_func, "add", backward_add);
    return C;
}

void backward_add(Tensor *C) {
    if (!C || !C->inputs || C->num_inputs < 2) return;
    
    Tensor *A = C->inputs[0];
    Tensor *B = C->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i];
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));

        if (A->ndim == 2 && B->ndim == 1 && A->shape[1] == B->shape[0]) {
            // also a temporary fix, should add broadcasting support properly
            for (size_t i = 0; i < A->shape[0]; i++) {
                for (size_t j = 0; j < B->shape[0]; j++) {
                    B->grad[j] += C->grad[i * A->shape[1] + j];
                }
            }
        } else {
            for (size_t i = 0; i < B->size; i++) {
                B->grad[i] += C->grad[i];
            }
        }
    }
}

Tensor* tensor_sub(Tensor *A, Tensor *B) {
    if (!A || !B) return NULL;
    
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    tensor_ewise(A, B, C, sub_func, "sub", backward_sub);
    return C;
}


void backward_sub(Tensor *C) {
    if (!C || !C->inputs || C->num_inputs < 2) return;
    
    Tensor *A = C->inputs[0];
    Tensor *B = C->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i];
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
        for (size_t i = 0; i < B->size; i++) {
            B->grad[i] -= C->grad[i];
        }
    }
}

Tensor* tensor_mul(Tensor *A, Tensor *B) {
    if (!A || !B) return NULL;
    
    Tensor *C = tensor_create(A->shape, A->ndim);
    if (!C) return NULL;

    tensor_ewise(A, B, C, mul_func, "mul", backward_mul);
    return C;
}

void backward_mul(Tensor *C) {
    if (!C || !C->inputs || C->num_inputs < 2) return;
    
    Tensor *A = C->inputs[0];
    Tensor *B = C->inputs[1];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->size; i++) {
            A->grad[i] += C->grad[i] * B->data[i];
        }
    }
    
    if (B->requires_grad) {
        if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
        for (size_t i = 0; i < B->size; i++) {
            B->grad[i] += C->grad[i] * A->data[i];
        }
    }
}

// ====================================================
// Linear Algebra
// ====================================================

Tensor* tensor_matmul(Tensor *A, Tensor *B) {
    if (!A || !B) return NULL;
    
    // Check for backend implementation via registry
    OpFn backend_fn = get_operation_fn("matmul");
    if (backend_fn) {
        return ((Tensor* (*)(Tensor*, Tensor*))backend_fn)(A, B);
    }
    
    // CPU fallback implementation
    if (A->ndim == 1 && B->ndim == 1) {
        if (A->shape[0] != B->shape[0]) return NULL;
        
        Tensor *C = tensor_create((size_t[]){1}, 1);
        if (!C) return NULL;
        
        float acc = 0.0f;
        for (size_t i = 0; i < A->shape[0]; i++) {
            acc += A->data[i] * B->data[i];
        }
        C->data[0] = acc;
        
        grad_update_two_vars(A, B, C, NULL, "matmul", backward_matmul);
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
        
        grad_update_two_vars(A, B, C, NULL, "matmul", backward_matmul);
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
        
        grad_update_two_vars(A, B, C, NULL, "matmul", backward_matmul);
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

        grad_update_two_vars(A, B, C, NULL, "matmul", backward_matmul);

        return C; 
    } else {
        return NULL; 
    }
}

void backward_matmul(Tensor *output) {
    if (!output || output->num_inputs != 2) return;
    
    Tensor *A = output->inputs[0];
    Tensor *B = output->inputs[1];
    
    if (A->ndim == 1 && B->ndim == 1) {
        if (A->requires_grad) {
            if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
            for (size_t i = 0; i < A->size; i++) {
                A->grad[i] += output->grad[0] * B->data[i];
            }
        }
        if (B->requires_grad) {
            if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
            for (size_t i = 0; i < B->size; i++) {
                B->grad[i] += output->grad[0] * A->data[i];
            }
        }
    }
    
    else if (A->ndim == 2 && B->ndim == 1) {
        if (A->requires_grad) {
            if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
            for (size_t i = 0; i < A->shape[0]; i++) {
                for (size_t j = 0; j < A->shape[1]; j++) {
                    A->grad[i * A->shape[1] + j] += output->grad[i] * B->data[j];
                }
            }
        }
        if (B->requires_grad) {
            if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
            for (size_t j = 0; j < B->shape[0]; j++) {
                float acc = 0.0f;
                for (size_t i = 0; i < A->shape[0]; i++) {
                    acc += A->data[i * A->shape[1] + j] * output->grad[i];
                }
                B->grad[j] += acc;
            }
        }
    }
    
    else if (A->ndim == 1 && B->ndim == 2) {
        if (A->requires_grad) {
            if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
            for (size_t i = 0; i < A->shape[0]; i++) {
                float acc = 0.0f;
                for (size_t j = 0; j < B->shape[1]; j++) {
                    acc += output->grad[j] * B->data[i * B->shape[1] + j];
                }
                A->grad[i] += acc;
            }
        }
        if (B->requires_grad) {
            if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
            for (size_t i = 0; i < B->shape[0]; i++) {
                for (size_t j = 0; j < B->shape[1]; j++) {
                    B->grad[i * B->shape[1] + j] += A->data[i] * output->grad[j];
                }
            }
        }
    }
    
    else if (A->ndim == 2 && B->ndim == 2) {
        if (A->requires_grad) {
            if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
            for (size_t i = 0; i < A->shape[0]; i++) {
                for (size_t j = 0; j < A->shape[1]; j++) {
                    float acc = 0.0f;
                    for (size_t k = 0; k < B->shape[1]; k++) {
                        acc += output->grad[i * output->shape[1] + k] * B->data[j * B->shape[1] + k];
                    }
                    A->grad[i * A->shape[1] + j] += acc;
                }
            }
        }
        if (B->requires_grad) {
            if (!B->grad) B->grad = (float *)calloc(B->size, sizeof(float));
            for (size_t i = 0; i < B->shape[0]; i++) {
                for (size_t j = 0; j < B->shape[1]; j++) {
                    float acc = 0.0f;
                    for (size_t k = 0; k < A->shape[0]; k++) {
                        acc += A->data[k * A->shape[1] + i] * output->grad[k * output->shape[1] + j];
                    }
                    B->grad[i * B->shape[1] + j] += acc;
                }
            }
        }
    }
}

Tensor* tensor_transpose2d(Tensor *A) {
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

    grad_update_one_var(A, C, NULL, "transpose2d", backward_transpose2d);

    return C; 
}

void backward_transpose2d(Tensor *C) {
    Tensor *A = C->inputs[0];
    
    if (A->requires_grad) {
        if (!A->grad) A->grad = (float *)calloc(A->size, sizeof(float));
        for (size_t i = 0; i < A->shape[0]; i++) {
            for (size_t j = 0; j < A->shape[1]; j++) {
                A->grad[i * A->shape[1] + j] += C->grad[j * A->shape[0] + i];
            }
        }
    }
}

// ====================================================
// Activation Functions
// ====================================================

Tensor* tensor_relu(Tensor *Z) {
    if (!Z) return NULL;

    // Check for backend implementation
    OpFn backend_fn = get_operation_fn("relu");
    if (backend_fn) {
        Tensor* result = ((Tensor* (*)(Tensor*))backend_fn)(Z);
        if (result) return result;
    }

    Tensor *A = tensor_create(Z->shape, Z->ndim); 
    if (!A) return NULL; 

    for (size_t i = 0; i < Z->size; i++) {
        A->data[i] = Z->data[i] > 0.0f ? Z->data[i] : 0.0f; 
    }

    grad_update_one_var(Z, A, NULL, "relu", backward_relu);

    return A; 
}

Tensor* tensor_sigmoid(Tensor *Z) {
    if (!Z) return NULL;

    // Check for backend implementation
    OpFn backend_fn = get_operation_fn("sigmoid");
    if (backend_fn) {
        Tensor* result = ((Tensor* (*)(Tensor*))backend_fn)(Z);
        if (result) return result;
    }

    Tensor *A = tensor_create(Z->shape, Z->ndim);
    if (!A) return NULL;

    for (size_t i = 0; i < Z->size; i++) {
        A->data[i] = 1.0f / (1.0f + expf(-Z->data[i])); 
    }

    grad_update_one_var(Z, A, NULL, "sigmoid", backward_sigmoid);

    return A;
}

void backward_relu(Tensor *A) {
    Tensor *Z = A->inputs[0];
    
    if (Z->requires_grad) {
        if (!Z->grad) Z->grad = (float *)calloc(Z->size, sizeof(float));
        for (size_t i = 0; i < Z->size; i++) {
            Z->grad[i] += A->grad[i] * (Z->data[i] > 0 ? 1.0f : 0.0f);
        }
    }
}

Tensor* tensor_tanh(Tensor *Z) {
    if (!Z) return NULL;

    // Check for backend implementation
    OpFn backend_fn = get_operation_fn("tanh");
    if (backend_fn) {
        Tensor* result = ((Tensor* (*)(Tensor*))backend_fn)(Z);
        if (result) return result;
    }

    Tensor *A = tensor_create(Z->shape, Z->ndim);
    if (!A) return NULL;

    for (size_t i = 0; i < Z->size; i++) {
        A->data[i] = tanhf(Z->data[i]); 
    }

    grad_update_one_var(Z, A, NULL, "tanh", backward_tanh);

    return A;
}

void backward_tanh(Tensor *A) {
    Tensor *Z = A->inputs[0];
    
    if (Z->requires_grad) {
        if (!Z->grad) Z->grad = (float *)calloc(Z->size, sizeof(float));
        for (size_t i = 0; i < Z->size; i++) {
            float t = A->data[i];
            Z->grad[i] += A->grad[i] * (1.0f - t * t);
        }
    }
}

void backward_sigmoid(Tensor *A) {
    Tensor *Z = A->inputs[0];
    
    if (Z->requires_grad) {
        if (!Z->grad) Z->grad = (float *)calloc(Z->size, sizeof(float));
        for (size_t i = 0; i < Z->size; i++) {
            float sig = A->data[i];
            Z->grad[i] += A->grad[i] * sig * (1.0f - sig);
        }
    }
}

Tensor* tensor_softmax(Tensor *Z) {
    if (!Z) return NULL;

    // Check for backend implementation
    OpFn backend_fn = get_operation_fn("softmax");
    if (backend_fn) {
        Tensor* result = ((Tensor* (*)(Tensor*))backend_fn)(Z);
        if (result) return result;
    }

    Tensor *A = tensor_create(Z->shape, Z->ndim);
    if (!A) return NULL; 

    size_t batch_size = (Z->ndim == 2) ? Z->shape[0] : 1;
    size_t num_classes = (Z->ndim == 2) ? Z->shape[1] : Z->size;

    for (size_t b = 0; b < batch_size; b++) {
        size_t offset = b * num_classes;
        
        float max_val = Z->data[offset];
        for (size_t i = 1; i < num_classes; i++) {
            if (Z->data[offset + i] > max_val) max_val = Z->data[offset + i];
        }

        float sum = 0.0f;
        for (size_t i = 0; i < num_classes; i++) {
            A->data[offset + i] = expf(Z->data[offset + i] - max_val);
            sum += A->data[offset + i];
        }

        for (size_t i = 0; i < num_classes; i++) {
            A->data[offset + i] /= sum;
        }
    }

    grad_update_one_var(Z, A, NULL, "softmax", backward_softmax);

    return A;
}

void backward_softmax(Tensor *A) {
    Tensor *Z = A->inputs[0];
    
    if (Z->requires_grad) {
        if (!Z->grad) Z->grad = (float *)calloc(Z->size, sizeof(float));
        
        size_t batch_size = (Z->ndim == 2) ? Z->shape[0] : 1;
        size_t num_classes = (Z->ndim == 2) ? Z->shape[1] : Z->size;

        for (size_t b = 0; b < batch_size; b++) {
            size_t offset = b * num_classes;
            for (size_t i = 0; i < num_classes; i++) {
                for (size_t j = 0; j < num_classes; j++) {
                    float delta = (i == j) ? 1.0f : 0.0f;
                    Z->grad[offset + i] += A->grad[offset + j] * A->data[offset + j] * (delta - A->data[offset + i]);
                }
            }
        }
    }
}

// ====================================================
// Loss Functions
// ====================================================

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
    
    if (predictions->requires_grad || targets->requires_grad) {
        loss->requires_grad = 1;
        loss->op_name = strdup("mse");
        loss->num_inputs = 2;
        loss->inputs = (Tensor **)malloc(2 * sizeof(Tensor *));
        loss->inputs[0] = predictions;
        loss->inputs[1] = targets;
        loss->backward_fn = backward_mse;
    }
    
    return loss;
}

void backward_mse(Tensor *L) {
    Tensor *predictions = L->inputs[0];
    Tensor *targets = L->inputs[1]; 

    if (predictions->requires_grad) {
        if (!predictions->grad) 
            predictions->grad = (float *)calloc(predictions->size, sizeof(float));
        for (size_t i = 0; i < predictions->size; i++) {
            predictions->grad[i] += 
                (2.0f / predictions->size) * (predictions->data[i] - targets->data[i]) * L->grad[0];
        }
    }

    if (targets->requires_grad) {
        if (!targets->grad) 
            targets->grad = (float *)calloc(targets->size, sizeof(float));
        for (size_t i = 0; i < targets->size; i++) {
            targets->grad[i] -= 
                (2.0f / targets->size) * (predictions->data[i] - targets->data[i]) * L->grad[0];
        }
    }
}

Tensor* tensor_cross_entropy(Tensor *predictions, Tensor *targets) {
    if (!check_pred_target(predictions, targets)) return NULL; 
    
    Tensor *loss = tensor_create((size_t[]){1}, 1);
    if (!loss) return NULL;

    float sum_ce_loss = 0.0f;
    float epsilon = 1e-7f;
    for (size_t i = 0; i < predictions->size; i++) {
        float pred = predictions->data[i];
        float target = targets->data[i];
        pred = pred < epsilon ? epsilon : (pred > 1.0f - epsilon ? 1.0f - epsilon : pred);
        sum_ce_loss += -target * logf(pred);
    }
    loss->data[0] = sum_ce_loss / predictions->size;
    
    if (predictions->requires_grad || targets->requires_grad) {
        loss->requires_grad = 1;
        loss->op_name = strdup("cross_entropy");
        loss->num_inputs = 2;
        loss->inputs = (Tensor **)malloc(2 * sizeof(Tensor *));
        loss->inputs[0] = predictions;
        loss->inputs[1] = targets;
        loss->backward_fn = backward_cross_entropy;
    }
    
    return loss;
}

void backward_cross_entropy(Tensor *L) {
    Tensor *predictions = L->inputs[0];
    Tensor *targets = L->inputs[1]; 
    float epsilon = 1e-7f;

    if (predictions->requires_grad) {
        if (!predictions->grad) 
            predictions->grad = (float *)calloc(predictions->size, sizeof(float));
        for (size_t i = 0; i < predictions->size; i++) {
            float pred = predictions->data[i];
            pred = pred < epsilon ? epsilon : (pred > 1.0f - epsilon ? 1.0f - epsilon : pred);
            predictions->grad[i] += 
                (-targets->data[i] / pred) * L->grad[0];
        }
    }

    if (targets->requires_grad) {
        if (!targets->grad) 
            targets->grad = (float *)calloc(targets->size, sizeof(float));
        for (size_t i = 0; i < targets->size; i++) {
            float pred = predictions->data[i];
            pred = pred < epsilon ? epsilon : pred;
            targets->grad[i] -= 
                ( -logf(pred) ) * L->grad[0];
        }
    }
}

Tensor* tensor_binary_cross_entropy(Tensor *predictions, Tensor *targets) {
    if (!check_pred_target(predictions, targets)) return NULL;
    
    Tensor *loss = tensor_create((size_t[]){1}, 1);
    if (!loss) return NULL;

    float sum_bce_loss = 0.0f;
    float epsilon = 1e-7f;
    for (size_t i = 0; i < predictions->size; i++) {
        float pred = predictions->data[i];
        float target = targets->data[i];
        pred = pred < epsilon ? epsilon : (pred > 1.0f - epsilon ? 1.0f - epsilon : pred);
        sum_bce_loss += -target * logf(pred) - (1.0f - target) * logf(1.0f - pred);
    }
    loss->data[0] = sum_bce_loss / predictions->size;
    
    if (predictions->requires_grad || targets->requires_grad) {
        loss->requires_grad = 1;
        loss->op_name = strdup("binary_cross_entropy");
        loss->num_inputs = 2;
        loss->inputs = (Tensor **)malloc(2 * sizeof(Tensor *));
        loss->inputs[0] = predictions;
        loss->inputs[1] = targets;
        loss->backward_fn = backward_binary_cross_entropy;
    }
    
    return loss;
}

void backward_binary_cross_entropy(Tensor *L) {
    Tensor *predictions = L->inputs[0];
    Tensor *targets = L->inputs[1]; 
    float epsilon = 1e-7f;

    if (predictions->requires_grad) {
        if (!predictions->grad) 
            predictions->grad = (float *)calloc(predictions->size, sizeof(float));
        for (size_t i = 0; i < predictions->size; i++) {
            float pred = predictions->data[i];
            pred = pred < epsilon ? epsilon : (pred > 1.0f - epsilon ? 1.0f - epsilon : pred);
            predictions->grad[i] += 
                (-(targets->data[i] / pred) + 
                 (1.0f - targets->data[i]) / (1.0f - pred)) * L->grad[0];
        }
    }

    if (targets->requires_grad) {
        if (!targets->grad) 
            targets->grad = (float *)calloc(targets->size, sizeof(float));
        for (size_t i = 0; i < targets->size; i++) {
            float pred = predictions->data[i];
            pred = pred < epsilon ? epsilon : (pred > 1.0f - epsilon ? 1.0f - epsilon : pred);
            targets->grad[i] -= 
                (-logf(pred) + -logf(1.0f - pred)) * L->grad[0];
        }
    }
}

// ====================================================
// Slice
// ====================================================

Tensor* tensor_slice(Tensor *input, size_t start, size_t end) {
    if (!input || start >= end || end > input->size) return NULL; 

    Tensor *slice = (Tensor*)malloc(sizeof(Tensor));
    if (!slice) return NULL;

    slice->ndim = input->ndim; 
    slice->shape = (size_t*)malloc(input->ndim * sizeof(size_t));
    if (!slice->shape) {
        free(slice);
        return NULL;
    }

    slice->shape[0] = end - start; 
    size_t stride = 1;
    for (size_t i = 1; i < input->ndim; i++) {
        slice->shape[i] = input->shape[i]; 
        stride *= input->shape[i]; 
    }

    slice->size = (end - start) * stride; 
    slice->data = input->data + (start * stride); 

    if (input->grad) {
        slice->grad = input->grad + (start * stride); 
    } else {
        slice->grad = NULL; 
    }

    slice->owns_data = 0; 
    
    slice->requires_grad = input->requires_grad; 

    slice->op_name = NULL; 
    slice->inputs = NULL; 
    slice->num_inputs = 0; 
    slice->backward_fn = NULL; 
    slice->extra_data = NULL; 

    return slice;
}

// ====================================================
// Operation Registration
// ====================================================

void ops_register_builtins(void) {
    register_loss("mse", tensor_mse);
    register_loss("cross_entropy", tensor_cross_entropy);
    register_loss("binary_cross_entropy", tensor_binary_cross_entropy);
    register_tensor_op("add", backward_add);
    register_tensor_op("sub", backward_sub);
    register_tensor_op("mul", backward_mul);
    register_tensor_op("matmul", backward_matmul);
    register_tensor_op("transpose2d", backward_transpose2d);
    register_tensor_op("relu", backward_relu);
    register_tensor_op("sigmoid", backward_sigmoid);
    register_tensor_op("tanh", backward_tanh);
    register_tensor_op("softmax", backward_softmax);
    register_tensor_op("mse", backward_mse);
    register_tensor_op("cross_entropy", backward_cross_entropy);
    register_tensor_op("binary_cross_entropy", backward_binary_cross_entropy);
}