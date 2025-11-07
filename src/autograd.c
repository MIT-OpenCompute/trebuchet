#include "basednn/autograd.h"
#include <stdlib.h>

// Tensor function gradients
void backward_add(Tensor *C) {
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

void backward_sub(Tensor *C) {
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

void backward_mul(Tensor *C) {
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

void backward_matmul(Tensor *output) {
    if (!output || output->num_inputs != 2) return;
    
    Tensor *A = output->inputs[0];
    Tensor *B = output->inputs[1];
    
    if (A->ndim == 1 && B->ndim == 1) {
        if (A->requires_grad) {
            for (size_t i = 0; i < A->size; i++) {
                A->grad[i] += output->grad[0] * B->data[i];
            }
        }
        if (B->requires_grad) {
            for (size_t i = 0; i < B->size; i++) {
                B->grad[i] += output->grad[0] * A->data[i];
            }
        }
    }
    
    else if (A->ndim == 2 && B->ndim == 1) {
        if (A->requires_grad) {
            for (size_t i = 0; i < A->shape[0]; i++) {
                for (size_t j = 0; j < A->shape[1]; j++) {
                    A->grad[i * A->shape[1] + j] += output->grad[i] * B->data[j];
                }
            }
        }
        if (B->requires_grad) {
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
            for (size_t i = 0; i < A->shape[0]; i++) {
                float acc = 0.0f;
                for (size_t j = 0; j < B->shape[1]; j++) {
                    acc += output->grad[j] * B->data[i * B->shape[1] + j];
                }
                A->grad[i] += acc;
            }
        }
        if (B->requires_grad) {
            for (size_t i = 0; i < B->shape[0]; i++) {
                for (size_t j = 0; j < B->shape[1]; j++) {
                    B->grad[i * B->shape[1] + j] += A->data[i] * output->grad[j];
                }
            }
        }
    }
    
    else if (A->ndim == 2 && B->ndim == 2) {
        if (A->requires_grad) {
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

void backward_transpose(Tensor *C) {
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

// Linear combination gradient
void backward_linear(Tensor *Z) {
    Tensor *W = Z->inputs[0];
    Tensor *X = Z->inputs[1];
    Tensor *b = Z->inputs[2];

    if (W->requires_grad) {
        if (!W->grad) W->grad = (float *)calloc(W->size, sizeof(float));
        for (size_t i = 0; i < W->shape[0]; i++) {
            for (size_t j = 0; j < W->shape[1]; j++) {
                for (size_t n = 0; n < X->shape[0]; n++) {
                    W->grad[i * W->shape[1] + j] += 
                        Z->grad[n * Z->shape[1] + j] * X->data[n * X->shape[1] + i];
                }
            }
        }
    }

    if (X->requires_grad) {
        if (!X->grad) X->grad = (float *)calloc(X->size, sizeof(float));
        for (size_t n = 0; n < X->shape[0]; n++) {
            for (size_t i = 0; i < X->shape[1]; i++) {
                for (size_t j = 0; j < W->shape[1]; j++) {
                    X->grad[n * X->shape[1] + i] += 
                        Z->grad[n * Z->shape[1] + j] * W->data[i * W->shape[1] + j];
                }
            }
        }
    }

    if (b->requires_grad) {
        if (!b->grad) b->grad = (float *)calloc(b->size, sizeof(float));
        for (size_t j = 0; j < b->shape[0]; j++) {
            for (size_t n = 0; n < X->shape[0]; n++) {
                b->grad[j] += Z->grad[n * Z->shape[1] + j];
            }
        }
    }
}

// Activation function gradients
void backward_relu(Tensor *A) {
    Tensor *Z = A->inputs[0];
    
    if (Z->requires_grad) {
        if (!Z->grad) Z->grad = (float *)calloc(Z->size, sizeof(float));
        for (size_t i = 0; i < Z->size; i++) {
            Z->grad[i] += A->grad[i] * (Z->data[i] > 0 ? 1.0f : 0.0f);
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

void backward_softmax(Tensor *A) {
    Tensor *Z = A->inputs[0];
    
    if (Z->requires_grad) {
        if (!Z->grad) Z->grad = (float *)calloc(Z->size, sizeof(float));
        for (size_t i = 0; i < Z->size; i++) {
            for (size_t j = 0; j < Z->size; j++) {
                float delta = (i == j) ? 1.0f : 0.0f;
                Z->grad[i] += A->grad[j] * A->data[j] * (delta - A->data[i]);
            }
        }
    }
}

// Loss function gradients
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

void backward_cross_entropy(Tensor *L) {
    Tensor *predictions = L->inputs[0];
    Tensor *targets = L->inputs[1]; 

    if (predictions->requires_grad) {
        if (!predictions->grad) 
            predictions->grad = (float *)calloc(predictions->size, sizeof(float));
        for (size_t i = 0; i < predictions->size; i++) {
            predictions->grad[i] += 
                (-targets->data[i] / predictions->data[i]) * L->grad[0];
        }
    }

    if (targets->requires_grad) {
        if (!targets->grad) 
            targets->grad = (float *)calloc(targets->size, sizeof(float));
        for (size_t i = 0; i < targets->size; i++) {
            targets->grad[i] -= 
                ( -logf(predictions->data[i]) ) * L->grad[0];
        }
    }
}

void backward_binary_cross_entropy(Tensor *L) {
    Tensor *predictions = L->inputs[0];
    Tensor *targets = L->inputs[1]; 

    if (predictions->requires_grad) {
        if (!predictions->grad) 
            predictions->grad = (float *)calloc(predictions->size, sizeof(float));
        for (size_t i = 0; i < predictions->size; i++) {
            predictions->grad[i] += 
                (-(targets->data[i] / predictions->data[i]) + 
                 (1.0f - targets->data[i]) / (1.0f - predictions->data[i])) * L->grad[0];
        }
    }

    if (targets->requires_grad) {
        if (!targets->grad) 
            targets->grad = (float *)calloc(targets->size, sizeof(float));
        for (size_t i = 0; i < targets->size; i++) {
            targets->grad[i] -= 
                (-logf(predictions->data[i]) + 
                 -logf(1.0f - predictions->data[i])) * L->grad[0];
        }
    }
}