#include "lessbasednn/network.h"
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define INITIAL_CAPACITY 8

// Network management
Network* network_create() {
    Network *net = (Network *)malloc(sizeof(Network)); 
    if (!net) return NULL;

    net->layers = (Layer**)malloc(INITIAL_CAPACITY * sizeof(Layer *));
    net->parameters = NULL; 
    net->num_layers = 0; 
    net->num_parameters = 0;
    net->capacity = INITIAL_CAPACITY;

    return net;
}

void network_add_layer(Network *net, Layer *layer) {
    if (!net || !layer) return;

    if (net->num_layers >= net->capacity) {
        net->capacity *= 2; 
        net->layers = (Layer **)realloc(net->layers, net->capacity * sizeof(Layer *));
    }

    net->layers[net->num_layers++] = layer;

    if (layer->num_parameters > 0) {
        for (size_t i = 0; i < layer->num_parameters; i++) {
            tensor_set_requires_grad(layer->parameters[i], 1);
        }
    }

    if (net->parameters) {
        free(net->parameters); 
    }
    net->parameters = network_get_parameters(net, &net->num_parameters);
}

void network_free(Network *net) {
    if (!net) return; 

    for (size_t i = 0; i < net->num_layers; i++) {
        layer_free(net->layers[i]);
    }

    free(net->layers);
    if (net->parameters) {
        free(net->parameters);
    }
    free(net);
}

// Forward pass
Tensor* network_forward(Network *net, Tensor *input) {
    if (!net || !input) return NULL; 

    Tensor *output = input; 

    for (size_t i = 0; i < net->num_layers; i++) {
        Tensor *new_output = layer_forward(net->layers[i], output); 
        output = new_output;
    }

    return output;
}

// Training
void network_train(Network *net, Optimizer *opt,  Tensor *input, Tensor *target, size_t epochs, size_t batch_size, LossType loss_type, int verbose) {
    if (!net || !opt || !input || !target) return; 

    size_t num_samples = input->shape[0]; 
    size_t num_batches = (num_samples + batch_size - 1) / batch_size; 

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f; 

        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t start = batch * batch_size; 
            size_t end = (start + batch_size < num_samples) ? (start + batch_size) : num_samples; 
            
            Tensor *batch_input = tensor_slice(input, start, end); 
            Tensor *batch_target = tensor_slice(target, start, end); 

            if (!batch_input || !batch_target) {
                if (batch_input) tensor_free(batch_input);
                if (batch_target) tensor_free(batch_target);
                continue; 
            }

            Tensor *predictions = network_forward(net, batch_input);
            if (!predictions) {
                tensor_free(batch_input);
                tensor_free(batch_target);
                continue; 
            }

            Tensor *loss_tensor = NULL;
            switch (loss_type) {
                case LOSS_MSE:
                    loss_tensor = tensor_mse(predictions, batch_target);
                    break;
                case LOSS_CROSS_ENTROPY:
                    loss_tensor = tensor_cross_entropy(predictions, batch_target);
                    break;
                case LOSS_BINARY_CROSS_ENTROPY:
                    loss_tensor = tensor_binary_cross_entropy(predictions, batch_target);
                    break;
                default:
                    tensor_free(batch_input);
                    tensor_free(batch_target);
                    tensor_free(predictions);
                    continue;
            }

            if (loss_tensor) {
                float loss = loss_tensor->data[0]; 
                total_loss += loss;

                network_zero_grad(net); 
                tensor_backward(loss_tensor); 

                optimizer_step(opt);

                tensor_free(loss_tensor);
            }

            tensor_free(batch_input); 
            tensor_free(batch_target);
            tensor_free(predictions);
        }

        if (verbose) printf("Epoch %zu/%zu, Loss: %.6f\n", epoch + 1, epochs, total_loss / num_batches);
    }
}

float network_train_step(Network *net, Tensor *input, Tensor *target, Optimizer *opt, LossType loss_type) {
    if (!net || !opt || !input || !target) return 0.0f;

    Tensor *predictions = network_forward(net, input);
    if (!predictions) return 0.0f;

    Tensor *loss_tensor = NULL;
    switch (loss_type) {
        case LOSS_MSE:
            loss_tensor = tensor_mse(predictions, target);
            break;
        case LOSS_CROSS_ENTROPY:
            loss_tensor = tensor_cross_entropy(predictions, target);
            break;
        case LOSS_BINARY_CROSS_ENTROPY:
            loss_tensor = tensor_binary_cross_entropy(predictions, target);
            break;
        default:
            tensor_free(predictions);
            return 0.0f;
    }

    if (!loss_tensor) {
        tensor_free(predictions);
        return 0.0f;
    }

    float loss = loss_tensor->data[0];
    
    network_zero_grad(net);
    tensor_backward(loss_tensor);
    optimizer_step(opt);

    tensor_free(loss_tensor);
    tensor_free(predictions);

    return loss;
}

void network_zero_grad(Network *net) {
    if (!net) return; 

    for (size_t i = 0; i < net->num_layers; i++) {
        layer_zero_grad(net->layers[i]);
    }
}

// Utilities
void network_print(Network *net) {
    if (!net) return; 

    printf("Network:\n");
    printf("Number of layers: %zu\n", net->num_layers);
    printf("Number of parameters: %zu\n", net->num_parameters);

    for (size_t i = 0; i < net->num_layers; i++) {
        Layer *layer = net->layers[i];
        printf("  Layer %zu: ", i + 1);
        
        switch (layer->type) {
            case LAYER_LINEAR:
                printf("Linear(%zu, %zu)\n", 
                       layer->weights->shape[0], layer->weights->shape[1]);
                break;
            case LAYER_RELU:
                printf("ReLU()\n");
                break;
            case LAYER_SIGMOID:
                printf("Sigmoid()\n");
                break;
            case LAYER_TANH:
                printf("Tanh()\n");
                break;
            case LAYER_SOFTMAX:
                printf("Softmax()\n");
                break;
            default:
                printf("Unknown\n");
        }
    }
}

Tensor** network_get_parameters(Network *net, size_t *num_params) {
    if (!net) {
        *num_params = 0;
        return NULL; 
    }

    size_t total = 0;
    for (size_t i = 0; i < net->num_layers; i++) {
        size_t count; 
        layer_get_parameters(net->layers[i], &count);
        total += count;
    }

    if (total == 0) {
        *num_params = 0;
        return NULL;
    }

    Tensor **params = (Tensor **)malloc(total * sizeof(Tensor *));
    if (!params) {
        *num_params = 0;
        return NULL;
    }

    size_t idx = 0; 
    for (size_t i = 0; i < net->num_layers; i++) {
        size_t count; 
        Tensor **layer_params = layer_get_parameters(net->layers[i], &count); 

        for (size_t j = 0; j < count; j++) {
            params[idx++] = layer_params[j]; 
        }
    }

    *num_params = total; 
    return params; 
}

float network_accuracy(Tensor *predictions, Tensor *targets) {
    if (!predictions || !targets) return 0.0f; 
    if (predictions->shape[0] != targets->shape[0]) return 0.0f;

    size_t num_samples = predictions->shape[0];
    size_t num_classes = predictions->shape[1];
    size_t correct = 0;

    for (size_t i = 0; i < num_samples; i++) {
        size_t pred_class = 0; 
        float max_pred = predictions->data[i * num_classes];
        for (size_t j = 1; j < num_classes; j++) {
            if (predictions->data[i * num_classes + j] > max_pred) {
                max_pred = predictions->data[i * num_classes + j];
                pred_class = j;
            }
        }
        size_t target_class = 0; 
        float max_target = targets->data[i * num_classes];
        for (size_t j = 1; j < num_classes; j++) {
            if (targets->data[i * num_classes + j] > max_target) {
                max_target = targets->data[i * num_classes + j];
                target_class = j;
            }
        }

        if (pred_class == target_class) {
            correct++;
        }
    }

    return (float)correct / num_samples;
}

static void layer_save(Layer *layer, FILE *file) {
    if (!layer || !file) return;

    fwrite(&layer->type, sizeof(LayerType), 1, file);

    switch (layer->type) {
        case LAYER_LINEAR:
            size_t in_features = layer->weights->shape[0];
            size_t out_features = layer->weights->shape[1];

            fwrite(&in_features, sizeof(size_t), 1, file);
            fwrite(&out_features, sizeof(size_t), 1, file);

            size_t weight_size = in_features * out_features;
            fwrite(layer->weights->data, sizeof(float), weight_size, file);
            fwrite(layer->bias->data, sizeof(float), out_features, file);
            break;

        case LAYER_RELU:
        case LAYER_SIGMOID:
        case LAYER_TANH:
        case LAYER_SOFTMAX:
            break;

        default:
            fprintf(stderr, "Warning: Unknown layer type %d\n", layer->type);
            break;
    }
}

void network_save(Network *net, const char *file_path) {
    if (!net || !file_path) return; 

    FILE *file = fopen(file_path, "wb"); 
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", file_path);
        return;
    }

    uint32_t magic_number = 0x42444E4E; // "bDDN"
    uint32_t version = 1;
    fwrite(&magic_number, sizeof(uint32_t), 1, file);
    fwrite(&version, sizeof(uint32_t), 1, file);

    fwrite(&net->num_layers, sizeof(size_t), 1, file);
    for (size_t i = 0; i < net->num_layers; i++) {
        layer_save(net->layers[i], file);
    }

    fclose(file);
    printf("Network saved to %s\n", file_path);
}

static Layer* layer_load(FILE *file) {
    if (!file) return NULL; 

    LayerType type; 
    if (fread(&type, sizeof(LayerType), 1, file) != 1) return NULL;

    Layer *layer = NULL; 
    switch (type) {
        case LAYER_LINEAR: {
            size_t in_features, out_features;
            if (fread(&in_features, sizeof(size_t), 1, file) != 1) return NULL;
            if (fread(&out_features, sizeof(size_t), 1, file) != 1) return NULL;

            layer = layer_create(LINEAR(in_features, out_features));
            if (!layer) return NULL;

            size_t weight_size = in_features * out_features;
            if (fread(layer->weights->data, sizeof(float), weight_size, file) != weight_size) {
                layer_free(layer);
                return NULL;
            }
            if (fread(layer->bias->data, sizeof(float), out_features, file) != out_features) {
                layer_free(layer);
                return NULL;
            }
            break;
        }
        case LAYER_RELU:
            layer = layer_create(RELU());
            break;
        case LAYER_SIGMOID:
            layer = layer_create(SIGMOID());
            break;
        case LAYER_TANH:
            layer = layer_create(TANH());
            break;
        case LAYER_SOFTMAX:
            layer = layer_create(SOFTMAX());
            break;
        default:
            fprintf(stderr, "Warning: Unknown layer type %d\n", type);
            return NULL;
    }
    return layer; 
}

Network* network_load(const char *file_path) {
    if (!file_path) return NULL; 

    FILE *file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "Error: Could not open file %s for reading\n", file_path);
        return NULL;
    }

    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1 || magic_number != 0x42444E4E) { // "bDDN"
        fprintf(stderr, "Error: Invalid file format for %s\n", file_path);
        fclose(file);
        return NULL;
    }

    uint32_t version; 
    if (fread(&version, sizeof(uint32_t), 1, file) != 1 || version != 1) {
        fprintf(stderr, "Error: Unsupported version %u in file %s\n", version, file_path);
        fclose(file);
        return NULL;
    }

    size_t num_layers;
    if (fread(&num_layers, sizeof(size_t), 1, file) != 1) {
        fprintf(stderr, "Error: Could not read number of layers from %s\n", file_path);
        fclose(file);
        return NULL;
    }

    Network *net = network_create();
    if (!net) {
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < num_layers; i++) {
        Layer *layer = layer_load(file); 
        if (!layer) {
            fprintf(stderr, "Error: Could not load layer %zu from %s\n", i, file_path);
            network_free(net);
            fclose(file);
            return NULL;
        }
        network_add_layer(net, layer); 
    }

    fclose(file);
    printf("Network loaded from %s\n", file_path);
    return net;
}