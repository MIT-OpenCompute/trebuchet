#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

typedef enum {
    LOSS_MSE,
    LOSS_CROSS_ENTROPY,
    LOSS_BINARY_CROSS_ENTROPY
} LossType;

// Training loss operations
Tensor* loss_mse(Tensor *predictions, Tensor *targets);
Tensor* loss_cross_entropy(Tensor *predictions, Tensor *targets);
Tensor* loss_binary_cross_entropy(Tensor *predictions, Tensor *targets);

// Inference loss operations
float loss_mse_value(Tensor *predictions, Tensor *targets);
float loss_cross_entropy_value(Tensor *predictions, Tensor *targets);
float loss_binary_cross_entropy_value(Tensor *predictions, Tensor *targets);

#endif