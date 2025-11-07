#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include "ops.h"

typedef struct Network Network;

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
} OptimizerType;

typedef struct {
    OptimizerType type; 
    union {
        struct {
            float learning_rate;
            float momentum;
            Tensor **velocity;
        } SGDState;
        struct {
            float learning_rate;
            float beta1;
            float beta2;
            float epsilon;
            int t;
            Tensor **m;
            Tensor **v;
        } AdamState;
    } params;
} OptimizerConfig; 

#define SGD(lr, momentum) (OptimizerConfig){ .type = OPTIMIZER_SGD, .params.SGDState = { lr, momentum, NULL } }
#define ADAM(lr, beta1, beta2, epsilon) (OptimizerConfig){ .type = OPTIMIZER_ADAM, .params.AdamState = { lr, beta1, beta2, epsilon, 0, NULL, NULL } }

typedef struct Optimizer Optimizer;

struct Optimizer {
    OptimizerType type;
    Tensor **parameters;
    size_t num_parameters;
    void (*step)(Optimizer *self);
    void (*zero_grad)(Optimizer *self);
    void (*free)(Optimizer *self);
    void *state;
};

// Optimizer constructor/destructors. 
Optimizer* optimizer_create(Network *net, OptimizerConfig config);

// Optimizer operations
void optimizer_step(Optimizer *opt);
void optimizer_zero_grad(Optimizer *opt);
void optimizer_free(Optimizer *opt); 

#endif