#include "../include/network.h"
#include "../include/registry.h"
#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>

#define INITIAL_CAPACITY 8

// ====================================================
// Network Management
// ====================================================

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

// ====================================================
// Forward
// ====================================================

Tensor* network_forward(Network *net, Tensor *input) {
    if (!net || !input) return NULL; 

    Tensor *output = input; 

    for (size_t i = 0; i < net->num_layers; i++) {
        Tensor *new_output = layer_forward(net->layers[i], output); 
        output = new_output;
    }

    return output;
}

// ====================================================
// Network Training
// ====================================================

void network_train(Network *net, Optimizer *opt,  Tensor *input, Tensor *target, size_t epochs, size_t batch_size, const char *loss_name, int verbose) {
    if (!net || !opt || !input || !target) return; 

    size_t num_samples = input->shape[0]; 
    size_t num_batches = (num_samples + batch_size - 1) / batch_size; 

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        #ifdef HAS_WGPU
        struct timeval tv_start, tv_end;
        gettimeofday(&tv_start, NULL);
        double forward_time = 0, backward_time = 0, optimizer_time = 0;
        #endif

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

            #ifdef HAS_WGPU
            struct timeval t1, t2, t3, t4;
            gettimeofday(&t1, NULL);
            #endif
            
            Tensor *predictions = network_forward(net, batch_input);
            if (!predictions) {
                tensor_free(batch_input);
                tensor_free(batch_target);
                continue; 
            }

            #ifdef HAS_WGPU
            gettimeofday(&t2, NULL);
            #endif

            LossFn loss_fn = get_loss_fn(loss_name);
            if (!loss_fn) {
                tensor_free(batch_input);
                tensor_free(batch_target);
                tensor_free(predictions);
                continue;
            }
            
            Tensor *loss_tensor = loss_fn(predictions, batch_target);

            if (loss_tensor) {
                float loss = loss_tensor->data[0]; 
                total_loss += loss;

                network_zero_grad(net);
                
                #ifdef HAS_WGPU
                gettimeofday(&t3, NULL);
                #endif
                
                tensor_backward(loss_tensor);
                
                #ifdef HAS_WGPU
                gettimeofday(&t4, NULL);
                forward_time += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
                backward_time += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
                #endif

                optimizer_step(opt);
                
                #ifdef HAS_WGPU
                struct timeval t5;
                gettimeofday(&t5, NULL);
                optimizer_time += (t5.tv_sec - t4.tv_sec) * 1000.0 + (t5.tv_usec - t4.tv_usec) / 1000.0;
                #endif

                tensor_free(loss_tensor);
            }

            tensor_free(batch_input); 
            tensor_free(batch_target);
            tensor_free(predictions);
        }

        #ifdef HAS_WGPU
        gettimeofday(&tv_end, NULL);
        double epoch_time = (tv_end.tv_sec - tv_start.tv_sec) * 1000.0 + 
                           (tv_end.tv_usec - tv_start.tv_usec) / 1000.0;
        double samples_per_sec = num_samples / (epoch_time / 1000.0);
        if (verbose) {
            printf("Epoch %zu/%zu, Loss: %.6f (%.0f ms total: fwd=%.0f, bwd=%.0f, opt=%.0f | %.1f samples/sec)\n", 
                   epoch + 1, epochs, total_loss / num_batches, epoch_time,
                   forward_time, backward_time, optimizer_time, samples_per_sec);
        }
        #else
        if (verbose) printf("Epoch %zu/%zu, Loss: %.6f\n", epoch + 1, epochs, total_loss / num_batches);
        #endif
    }
}

float network_train_step(Network *net, Tensor *input, Tensor *target, Optimizer *opt, const char *loss_name) {
    if (!net || !opt || !input || !target) return 0.0f;

    Tensor *predictions = network_forward(net, input);
    if (!predictions) return 0.0f;

    LossFn loss_fn = get_loss_fn(loss_name);
    if (!loss_fn) {
        tensor_free(predictions);
        return 0.0f;
    }
    
    Tensor *loss_tensor = loss_fn(predictions, target);

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

// ====================================================
// Utilities
// ====================================================

void network_print(Network *net) {
    if (!net) return; 

    printf("Network:\n");
    printf("Number of layers: %zu\n", net->num_layers);
    printf("Number of parameters: %zu\n", net->num_parameters);

    for (size_t i = 0; i < net->num_layers; i++) {
        Layer *layer = net->layers[i];
        printf("  Layer %zu: ", i + 1);
        
        if (strcmp(layer->name, "linear") == 0) {
            printf("Linear(%zu, %zu)\n", 
                   layer->weights->shape[0], layer->weights->shape[1]);
        } else {
            printf("%s()\n", layer->name);
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

// ====================================================
// Save/Load
// ====================================================

static void layer_save(Layer *layer, FILE *file) {
    if (!layer || !file) return;

    size_t name_len = strlen(layer->name) + 1;
    fwrite(&name_len, sizeof(size_t), 1, file);
    fwrite(layer->name, sizeof(char), name_len, file);

    fwrite(&layer->config_data_size, sizeof(size_t), 1, file);
    if (layer->config_data_size > 0 && layer->config_data) {
        fwrite(layer->config_data, 1, layer->config_data_size, file);
    }

    fwrite(&layer->num_parameters, sizeof(size_t), 1, file);
    
    for (size_t i = 0; i < layer->num_parameters; i++) {
        Tensor *param = layer->parameters[i];
        fwrite(&param->ndim, sizeof(size_t), 1, file);
        fwrite(param->shape, sizeof(size_t), param->ndim, file);
        fwrite(param->data, sizeof(float), param->size, file);
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

    size_t name_len;
    if (fread(&name_len, sizeof(size_t), 1, file) != 1) return NULL;
    
    char *name = malloc(name_len);
    if (fread(name, sizeof(char), name_len, file) != name_len) {
        free(name);
        return NULL;
    }

    // Load config data
    size_t config_data_size;
    if (fread(&config_data_size, sizeof(size_t), 1, file) != 1) {
        free(name);
        return NULL;
    }
    
    void *config_data = NULL;
    if (config_data_size > 0) {
        config_data = malloc(config_data_size);
        if (fread(config_data, 1, config_data_size, file) != config_data_size) {
            free(config_data);
            free(name);
            return NULL;
        }
    }

    LayerConfig config = {.name = name, .params = config_data};
    Layer *layer = layer_create(config);
    if (!layer) {
        if (config_data) free(config_data);
        free(name);
        return NULL;
    }

    size_t num_params;
    if (fread(&num_params, sizeof(size_t), 1, file) != 1) {
        layer_free(layer);
        if (config_data) free(config_data);
        free(name);
        return NULL;
    }

    for (size_t i = 0; i < num_params; i++) {
        if (i >= layer->num_parameters) {
            fprintf(stderr, "Error: Parameter count mismatch for layer %s\n", name);
            layer_free(layer);
            if (config_data) free(config_data);
            free(name);
            return NULL;
        }
        
        Tensor *param = layer->parameters[i];
        size_t ndim;
        if (fread(&ndim, sizeof(size_t), 1, file) != 1) {
            layer_free(layer);
            if (config_data) free(config_data);
            free(name);
            return NULL;
        }
        
        size_t *shape = malloc(ndim * sizeof(size_t));
        if (fread(shape, sizeof(size_t), ndim, file) != ndim) {
            free(shape);
            layer_free(layer);
            if (config_data) free(config_data);
            free(name);
            return NULL;
        }
        
        size_t size = 1;
        for (size_t j = 0; j < ndim; j++) {
            size *= shape[j];
        }
        free(shape);
        
        if (fread(param->data, sizeof(float), size, file) != size) {
            layer_free(layer);
            if (config_data) free(config_data);
            free(name);
            return NULL;
        }
    }
    
    if (config_data) free(config_data);
    free(name);
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