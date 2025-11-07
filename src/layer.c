#include "basednn/layer.h"
#include <stdlib.h>

// Forward forward (lol) declarations
static Tensor* linear_forward(Layer *self, Tensor *input); 
static Tensor* relu_forward(Layer *self, Tensor *input);
static Tensor* sigmoid_forward(Layer *self, Tensor *input);
static Tensor* tanh_forward(Layer *self, Tensor *input);
static Tensor* softmax_forward(Layer *self, Tensor *input);

Layer* layer_create(LayerConfig config) {
    Layer *layer = malloc(sizeof(Layer)); 
    layer->type = config.type; 
    layer->weights = NULL;
    layer->bias = NULL;
    layer->output = NULL;
    layer->parameters = NULL;
    layer->num_parameters = 0;

    switch (config.type) {
        case LAYER_LINEAR: {
            size_t in_features = config.params.linear.in_features;
            size_t out_features = config.params.linear.out_features; 

            layer->weights = tensor_randn(2, (size_t[]){in_features, out_features}, 42);
            layer->bias = tensor_zeroes(1, (size_t[]){out_features});

            layer->parameters = malloc(2 * sizeof(Tensor*));
            layer->parameters[0] = layer->weights;
            layer->parameters[1] = layer->bias;
            layer->num_parameters = 2;

            layer->forward = linear_forward;
            break;
        } case LAYER_RELU:
            layer->forward = relu_forward;
            break;
        case LAYER_SIGMOID:
            layer->forward = sigmoid_forward;
            break;
        case LAYER_TANH:
            layer->forward = tanh_forward;
            break;
        case LAYER_SOFTMAX:
            layer->forward = softmax_forward;
            break;
        default:
            free(layer);
            return NULL;
    }
    return layer;
}

void layer_free(Layer *layer) {
    if (!layer) return; 

    if (layer->weights) tensor_free(layer->weights);
    if (layer->bias) tensor_free(layer->bias);
    if (layer->output) tensor_free(layer->output);

    free(layer);
}

Tensor* layer_forward(Layer *layer, Tensor *input) {
    if (!layer || !layer->forward) return NULL; 
    return layer->forward(layer, input);
}

// Layer forward implementations
static Tensor* linear_forward(Layer *self, Tensor *input) {
    if (!self || !input || !self->weights || !self->bias) return NULL; 

    Tensor *Z_0 = tensor_matmul(input, self->weights);
    Tensor *Z = tensor_add(Z_0, self->bias);

    return Z;
}

static Tensor* relu_forward(Layer *self, Tensor *input) {
    return tensor_relu(input);
}

static Tensor* sigmoid_forward(Layer *self, Tensor *input) {
    return tensor_sigmoid(input);
}

static Tensor* tanh_forward(Layer *self, Tensor *input) {
    return tensor_tanh(input);
}

static Tensor* softmax_forward(Layer *self, Tensor *input) {
    return tensor_softmax(input);
}

// Utilities
void layer_zero_grad(Layer *layer) {
    if (!layer) return;

    for (size_t i = 0; i < layer->num_parameters; i++) {
        if (layer->parameters[i]->grad) {
            tensor_zero_grad(layer->parameters[i]);
        }
    }
}

Tensor** layer_get_parameters(Layer *layer, size_t *num_params) {
    if (!layer || !num_params) return NULL;

    *num_params = layer->num_parameters;
    return layer->parameters;
}