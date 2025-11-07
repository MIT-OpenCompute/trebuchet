#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include "layer.h"

typedef enum {
    OPTIMIZER_SGD, 
    OPTIMIZER_ADAM
} OptimizerType;

typedef struct {
    OptimizerType type; 
    float learning_rate; 
    void (*step)(struct Optimizer *self, Layer **layers, size_t num_layers);
    void (*zero_grad)(struct Optimizer *self, Layer **layers, size_t num_layers); 
    void (*free)(struct Optimizer *self); 
} Optimizer;

typedef struct {
    Optimizer base;
    float momentum; 
    Tensor **velocity; 
} SGDOptimizer; 

// SGD optimizer constructor
SGDOptimizer* optimizer_sgd_create(float learning_rate, float momentum); 

typedef struct {
    Optimizer base; 
    float beta1; 
    float beta2; 
    float epsilon; 
    int t;
    Tensor **m; 
    Tensor **v;
} AdamOptimizer;

// Adam optimizer constructor
AdamOptimizer* optimizer_adam_create(float learning_rate, float beta1, float beta2, float epsilon); 

// Optimizer operations
void optimizer_step(Optimizer *opt, Layer **layers, size_t num_layers);
void optimizer_zero_grad(Optimizer *opt, Layer **layers, size_t num_layers); 
void optimizer_free(Optimizer *opt); 

#endif