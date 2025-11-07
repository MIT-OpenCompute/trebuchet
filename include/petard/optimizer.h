#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include "layer.h"
#include "network.h"

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_RMSPROP
} OptimizerType;

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

typedef struct {
    float learning_rate;
    float momentum;
    Tensor **velocity;
} SGDState;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    Tensor **m;
    Tensor **v;
} AdamState;

// Layer optimizer constructors
Optimizer* optimizer_sgd_from_layers(Layer **layers, size_t num_layers, float learning_rate, float momentum);
Optimizer* optimizer_adam_from_layers(Layer **layers, size_t num_layers, float learning_rate, float beta1, float beta2, float epsilon);
Optimizer* optimizer_sgd_from_network(Network *net, float learning_rate, float momentum);
Optimizer* optimizer_adam_from_network(Network *net, float learning_rate, float beta1, float beta2, float epsilon);
Optimizer* optimizer_sgd_create(Tensor **parameters, size_t num_parameters, float learning_rate, float momentum);
Optimizer* optimizer_adam_create(Tensor **parameters, size_t num_parameters, float learning_rate, float beta1, float beta2, float epsilon);

// Optimizer operations
void optimizer_step(Optimizer *opt);
void optimizer_zero_grad(Optimizer *opt);
void optimizer_free(Optimizer *opt); 

#endif