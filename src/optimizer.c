#include "lessbasednn/optimizer.h"
#include "lessbasednn/network.h"
#include <stdlib.h>
#include <string.h> 
#include <math.h> 

static void sgd_step(Optimizer *opt);
static void adam_step(Optimizer *opt);

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

// Helper tensor functions
static void tensor_scale(Tensor *T, float scale) {
    if (!T || !T->data) return; 
    for (size_t i = 0; i < T->size; i++) {
        T->data[i] *= scale; 
    }
}

static void tensor_add_inplace(Tensor *dst, Tensor *src) {
    if (!dst || !src || !dst->data || !src->data) return; 
    for (size_t i = 0; i < dst->size; i++) {
        dst->data[i] += src->data[i]; 
    }
}

static void tensor_scale_add(Tensor *dst, Tensor *src, float scale) {
    tensor_scale(src, scale);
    tensor_add_inplace(dst, src);
}

// Optimizer constructor/destuctors
Optimizer* optimizer_create(Network *net, OptimizerConfig config) {
    if (!net) return NULL; 

    Optimizer *opt = (Optimizer *)malloc(sizeof(Optimizer)); 
    if (!opt) return NULL; 

    opt->type = config.type; 
    opt->parameters = net->parameters; 
    opt->num_parameters = net->num_parameters;
    opt->zero_grad = optimizer_zero_grad;

    if (config.type == OPTIMIZER_SGD) {
        opt->step = sgd_step; 
        opt->free = optimizer_free; 

        SGDState *state = (SGDState *)malloc(sizeof(SGDState));
        if (!state) {
            free(opt); 
            return NULL; 
        }

        state->learning_rate = config.params.SGDState.learning_rate;
        state->momentum = config.params.SGDState.momentum;
        state->velocity = NULL; 

        if (state->momentum > 0.0f) {
            state->velocity = (Tensor **)malloc(opt->num_parameters * sizeof(Tensor *));
            if (!state->velocity) {
                free(state);
                free(opt);
                return NULL;
            }

            for (size_t i = 0; i < opt->num_parameters; i++) {
                state->velocity[i] = tensor_create(opt->parameters[i]->shape, opt->parameters[i]->ndim);
                
                if (!state->velocity[i]) {
                    for (size_t j = 0; j < i; j++) {
                        tensor_free(state->velocity[j]);
                    }
                    free(state->velocity);
                    free(state);
                    free(opt);
                    return NULL;
                }
                tensor_fill(state->velocity[i], 0.0f);
            }
        }
        opt->state = state; 

    } else if (config.type == OPTIMIZER_ADAM) {
        opt->step = adam_step; 
        opt->free = optimizer_free; 

        AdamState *state = (AdamState *)malloc(sizeof(AdamState));
        if (!state) {
            free(opt);
            return NULL;
        }

        state->learning_rate = config.params.AdamState.learning_rate;
        state->beta1 = config.params.AdamState.beta1;
        state->beta2 = config.params.AdamState.beta2;
        state->epsilon = config.params.AdamState.epsilon;
        state->t = 0;

        state->m = (Tensor **)malloc(opt->num_parameters * sizeof(Tensor *));
        state->v = (Tensor **)malloc(opt->num_parameters * sizeof(Tensor *));

        if (!state->m || !state->v) {
            free(state->m);
            free(state->v);
            free(state);
            free(opt);
            return NULL;
        }

        for (size_t i = 0; i < opt->num_parameters; i++) {
            state->m[i] = tensor_create(opt->parameters[i]->shape, opt->parameters[i]->ndim);
            state->v[i] = tensor_create(opt->parameters[i]->shape, opt->parameters[i]->ndim);

            if (!state->m[i] || !state->v[i]) {
                for (size_t j = 0; j < i; j++) {
                    tensor_free(state->m[j]);
                    tensor_free(state->v[j]);
                }
                free(state->m);
                free(state->v);
                free(state);
                free(opt);
                return NULL;
            }
            tensor_fill(state->m[i], 0.0f);
            tensor_fill(state->v[i], 0.0f);
        }
        opt->state = state;
    }
    return opt; 
}

static void sgd_step(Optimizer *opt) {
    if (!opt) return; 

    SGDState *state = (SGDState *)opt->state; 

    for (size_t i = 0; i < opt->num_parameters; i++) {
        Tensor *param = opt->parameters[i]; 
        if (!param->grad) continue;

        if (state->momentum > 0.0f) {
            tensor_scale(state->velocity[i], state->momentum);
            for (size_t j = 0; j < param->size; j++) {
                state->velocity[i]->data[j] += -state->learning_rate * param->grad[j];
                param->data[j] += state->velocity[i]->data[j]; 
            }
        } else {
            for (size_t j = 0; j < param->size; j++) {
                param->data[j] += -state->learning_rate * param->grad[j];
            }
        }
    }
}

static void adam_step(Optimizer *opt) {
    if (!opt) return; 

    AdamState *state = (AdamState *)opt->state; 
    state->t += 1; 

    float bias_correction1 = 1.0f - powf(state->beta1, state->t);
    float bias_correction2 = 1.0f - powf(state->beta2, state->t);

    for (size_t i = 0; i < opt->num_parameters; i++) {
        Tensor *param = opt->parameters[i]; 
        if (!param->grad) continue; 

        for (size_t j = 0; j < param->size; j++) {
            state->m[i]->data[j] = state->beta1 * state->m[i]->data[j] + (1.0f - state->beta1) * param->grad[j];
            state->v[i]->data[j] = state->beta2 * state->v[i]->data[j] + (1.0f - state->beta2) * param->grad[j] * param->grad[j];
            
            float m_hat = state->m[i]->data[j] / bias_correction1;
            float v_hat = state->v[i]->data[j] / bias_correction2;
            param->data[j] -= state->learning_rate * m_hat / (sqrtf(v_hat) + state->epsilon);
        }
    }
}

void optimizer_step(Optimizer *opt) {
    if (!opt || !opt->step) return;
    opt->step(opt);
}

void optimizer_zero_grad(Optimizer *opt) {
    if (!opt) return; 

    for (size_t i = 0; i < opt->num_parameters; i++) {
        Tensor *param = opt->parameters[i]; 
        if (param->grad) {
            tensor_zero_grad(param);
        }
    }
}

static void sgd_free_state(SGDState *state, size_t num_parameters) {
    if (!state) return;
    
    if (state->velocity) {
        for (size_t i = 0; i < num_parameters; i++) {
            if (state->velocity[i]) {
                tensor_free(state->velocity[i]);
            }
        }
        free(state->velocity);
    }
    free(state);
}

static void adam_free_state(AdamState *state, size_t num_parameters) {
    if (!state) return;
    
    if (state->m) {
        for (size_t i = 0; i < num_parameters; i++) {
            if (state->m[i]) {
                tensor_free(state->m[i]);
            }
        }
        free(state->m);
    }
    if (state->v) {
        for (size_t i = 0; i < num_parameters; i++) {
            if (state->v[i]) {
                tensor_free(state->v[i]);
            }
        }
        free(state->v);
    }
    free(state);
}

void optimizer_free(Optimizer *opt) {
    if (!opt) return; 

    if (opt->state) {
        if (opt->type == OPTIMIZER_SGD) {
            sgd_free_state((SGDState *)opt->state, opt->num_parameters);
        } else if (opt->type == OPTIMIZER_ADAM) {
            adam_free_state((AdamState *)opt->state, opt->num_parameters);
        } else {
            free(opt->state); 
        }
    }
    free(opt);
}