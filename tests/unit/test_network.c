/*
 * AI generated Unit tests for network.c
 * Tests network creation, layer management, forward pass, training, and utilities
 */

#include "lessbasednn/network.h"
#include "lessbasednn/layer.h"
#include "lessbasednn/optimizer.h"
#include "lessbasednn/tensor.h"
#include "lessbasednn/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-5
#define ASSERT_FLOAT_EQ(a, b) assert(fabsf((a) - (b)) < EPSILON)
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    printf("Running test_%s...\n", #name); \
    test_##name(); \
    printf("  ✓ test_%s passed\n", #name); \
} while(0)

// ============ Network Creation Tests ============

TEST(network_create_basic) {
    Network *net = network_create();
    
    assert(net != NULL);
    assert(net->layers != NULL);
    assert(net->num_layers == 0);
    assert(net->num_parameters == 0);
    assert(net->capacity == 8);  // Initial capacity
    
    network_free(net);
}

TEST(network_add_single_layer) {
    Network *net = network_create();
    
    LayerConfig config = LINEAR(10, 5);
    Layer *layer = layer_create(config);
    
    network_add_layer(net, layer);
    
    assert(net->num_layers == 1);
    assert(net->layers[0] == layer);
    assert(net->num_parameters > 0);
    
    network_free(net);
}

TEST(network_add_multiple_layers) {
    Network *net = network_create();
    
    Layer *layer1 = layer_create(LINEAR(10, 5));
    Layer *layer2 = layer_create(RELU());
    Layer *layer3 = layer_create(LINEAR(5, 2));
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    
    assert(net->num_layers == 3);
    assert(net->layers[0] == layer1);
    assert(net->layers[1] == layer2);
    assert(net->layers[2] == layer3);
    
    network_free(net);
}

TEST(network_capacity_expansion) {
    Network *net = network_create();
    
    // Add more layers than initial capacity
    for (int i = 0; i < 10; i++) {
        Layer *layer = layer_create(RELU());
        network_add_layer(net, layer);
    }
    
    assert(net->num_layers == 10);
    assert(net->capacity >= 10);
    
    network_free(net);
}

// ============ Forward Pass Tests ============

TEST(network_forward_single_layer) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(4, 3));
    tensor_fill(layer->weights, 1.0f);
    tensor_fill(layer->bias, 0.5f);
    network_add_layer(net, layer);
    
    size_t input_shape[] = {4};
    Tensor *input = tensor_ones(input_shape, 1);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 3);
    
    // [1, 1, 1, 1] @ weights (all 1) + bias (0.5) = [4.5, 4.5, 4.5]
    for (size_t i = 0; i < output->size; i++) {
        ASSERT_FLOAT_EQ(output->data[i], 4.5f);
    }
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_forward_multi_layer) {
    Network *net = network_create();
    
    // Create 3-layer network: Linear(5->4) -> ReLU -> Linear(4->3)
    Layer *layer1 = layer_create(LINEAR(5, 4));
    Layer *layer2 = layer_create(RELU());
    Layer *layer3 = layer_create(LINEAR(4, 3));
    
    tensor_fill(layer1->weights, 0.5f);
    tensor_fill(layer1->bias, 0.0f);
    tensor_fill(layer3->weights, 0.5f);
    tensor_fill(layer3->bias, 0.0f);
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    
    size_t input_shape[] = {5};
    Tensor *input = tensor_ones(input_shape, 1);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 3);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_forward_batch) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(4, 2));
    tensor_fill(layer->weights, 1.0f);
    tensor_fill(layer->bias, 0.0f);
    network_add_layer(net, layer);
    
    // Batch of 3 samples
    size_t input_shape[] = {3, 4};
    Tensor *input = tensor_ones(input_shape, 2);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->ndim == 2);
    assert(output->shape[0] == 3);
    assert(output->shape[1] == 2);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_forward_with_activations) {
    Network *net = network_create();
    
    Layer *linear = layer_create(LINEAR(3, 3));
    Layer *sigmoid = layer_create(SIGMOID());
    
    tensor_fill(linear->weights, 0.5f);
    tensor_fill(linear->bias, 0.0f);
    
    network_add_layer(net, linear);
    network_add_layer(net, sigmoid);
    
    size_t input_shape[] = {3};
    Tensor *input = tensor_ones(input_shape, 1);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    // Output should be sigmoid activated (between 0 and 1)
    for (size_t i = 0; i < output->size; i++) {
        assert(output->data[i] > 0.0f && output->data[i] < 1.0f);
    }
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

// ============ Parameter Management Tests ============

TEST(network_get_parameters_empty) {
    Network *net = network_create();
    
    size_t num_params;
    Tensor **params = network_get_parameters(net, &num_params);
    
    assert(num_params == 0);
    assert(params == NULL);
    
    network_free(net);
}

TEST(network_get_parameters_single_layer) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(3, 2));
    network_add_layer(net, layer);
    
    size_t num_params;
    Tensor **params = network_get_parameters(net, &num_params);
    
    assert(params != NULL);
    assert(num_params == 2);  // Weights and bias
    assert(params[0] == layer->weights);
    assert(params[1] == layer->bias);
    
    network_free(net);
}

TEST(network_get_parameters_multiple_layers) {
    Network *net = network_create();
    
    Layer *layer1 = layer_create(LINEAR(5, 4));
    Layer *layer2 = layer_create(RELU());
    Layer *layer3 = layer_create(LINEAR(4, 3));
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    
    size_t num_params;
    Tensor **params = network_get_parameters(net, &num_params);
    
    assert(params != NULL);
    assert(num_params == 4);  // 2 from layer1 + 0 from layer2 + 2 from layer3
    
    network_free(net);
}

TEST(network_zero_grad) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(3, 2));
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    
    // Allocate and fill gradients
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    layer->bias->grad = (float*)calloc(layer->bias->size, sizeof(float));
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 1.0f;
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        layer->bias->grad[i] = 2.0f;
    }
    
    network_add_layer(net, layer);
    
    network_zero_grad(net);
    
    // All gradients should be zero
    for (size_t i = 0; i < layer->weights->size; i++) {
        ASSERT_FLOAT_EQ(layer->weights->grad[i], 0.0f);
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        ASSERT_FLOAT_EQ(layer->bias->grad[i], 0.0f);
    }
    
    network_free(net);
}

// ============ Accuracy Function Tests ============

TEST(network_accuracy_perfect) {
    size_t shape[] = {3, 2};
    Tensor *predictions = tensor_create(shape, 2);
    Tensor *targets = tensor_create(shape, 2);
    
    // Sample 0: class 0
    predictions->data[0] = 0.9f; predictions->data[1] = 0.1f;
    targets->data[0] = 1.0f; targets->data[1] = 0.0f;
    
    // Sample 1: class 1
    predictions->data[2] = 0.2f; predictions->data[3] = 0.8f;
    targets->data[2] = 0.0f; targets->data[3] = 1.0f;
    
    // Sample 2: class 0
    predictions->data[4] = 0.7f; predictions->data[5] = 0.3f;
    targets->data[4] = 1.0f; targets->data[5] = 0.0f;
    
    float acc = network_accuracy(predictions, targets);
    ASSERT_FLOAT_EQ(acc, 1.0f);  // 100% accuracy
    
    tensor_free(predictions);
    tensor_free(targets);
}

TEST(network_accuracy_partial) {
    size_t shape[] = {4, 3};
    Tensor *predictions = tensor_create(shape, 2);
    Tensor *targets = tensor_create(shape, 2);
    
    // Sample 0: predicted class 0, target class 0 (correct)
    predictions->data[0] = 0.8f; predictions->data[1] = 0.1f; predictions->data[2] = 0.1f;
    targets->data[0] = 1.0f; targets->data[1] = 0.0f; targets->data[2] = 0.0f;
    
    // Sample 1: predicted class 1, target class 2 (wrong)
    predictions->data[3] = 0.1f; predictions->data[4] = 0.7f; predictions->data[5] = 0.2f;
    targets->data[3] = 0.0f; targets->data[4] = 0.0f; targets->data[5] = 1.0f;
    
    // Sample 2: predicted class 2, target class 2 (correct)
    predictions->data[6] = 0.1f; predictions->data[7] = 0.2f; predictions->data[8] = 0.7f;
    targets->data[6] = 0.0f; targets->data[7] = 0.0f; targets->data[8] = 1.0f;
    
    // Sample 3: predicted class 0, target class 0 (correct)
    predictions->data[9] = 0.9f; predictions->data[10] = 0.05f; predictions->data[11] = 0.05f;
    targets->data[9] = 1.0f; targets->data[10] = 0.0f; targets->data[11] = 0.0f;
    
    float acc = network_accuracy(predictions, targets);
    ASSERT_FLOAT_EQ(acc, 0.75f);  // 75% accuracy (3 out of 4 correct)
    
    tensor_free(predictions);
    tensor_free(targets);
}

TEST(network_accuracy_zero) {
    size_t shape[] = {2, 2};
    Tensor *predictions = tensor_create(shape, 2);
    Tensor *targets = tensor_create(shape, 2);
    
    // Sample 0: predicted class 0, target class 1 (wrong)
    predictions->data[0] = 0.9f; predictions->data[1] = 0.1f;
    targets->data[0] = 0.0f; targets->data[1] = 1.0f;
    
    // Sample 1: predicted class 0, target class 1 (wrong)
    predictions->data[2] = 0.8f; predictions->data[3] = 0.2f;
    targets->data[2] = 0.0f; targets->data[3] = 1.0f;
    
    float acc = network_accuracy(predictions, targets);
    ASSERT_FLOAT_EQ(acc, 0.0f);  // 0% accuracy
    
    tensor_free(predictions);
    tensor_free(targets);
}

// ============ Edge Cases and Error Handling ============

TEST(network_forward_null_input) {
    Network *net = network_create();
    Layer *layer = layer_create(LINEAR(3, 2));
    network_add_layer(net, layer);
    
    Tensor *output = network_forward(net, NULL);
    assert(output == NULL);
    
    network_free(net);
}

TEST(network_forward_null_network) {
    size_t shape[] = {3};
    Tensor *input = tensor_ones(shape, 1);
    
    Tensor *output = network_forward(NULL, input);
    assert(output == NULL);
    
    tensor_free(input);
}

TEST(network_forward_empty_network) {
    Network *net = network_create();
    
    size_t shape[] = {3};
    Tensor *input = tensor_ones(shape, 1);
    
    Tensor *output = network_forward(net, input);
    
    // Should return input if no layers
    assert(output == input);
    
    tensor_free(input);
    network_free(net);
}

TEST(network_free_null) {
    network_free(NULL);  // Should not crash
}

TEST(network_add_layer_null_network) {
    Layer *layer = layer_create(LINEAR(3, 2));
    network_add_layer(NULL, layer);  // Should not crash
    layer_free(layer);
}

TEST(network_add_layer_null_layer) {
    Network *net = network_create();
    network_add_layer(net, NULL);  // Should not crash
    assert(net->num_layers == 0);
    network_free(net);
}

TEST(network_zero_grad_null) {
    network_zero_grad(NULL);  // Should not crash
}

TEST(network_accuracy_null_inputs) {
    size_t shape[] = {2, 2};
    Tensor *t = tensor_ones(shape, 2);
    
    assert(network_accuracy(NULL, t) == 0.0f);
    assert(network_accuracy(t, NULL) == 0.0f);
    assert(network_accuracy(NULL, NULL) == 0.0f);
    
    tensor_free(t);
}

TEST(network_accuracy_mismatched_shapes) {
    size_t shape_a[] = {3, 2};
    size_t shape_b[] = {2, 2};
    
    Tensor *a = tensor_ones(shape_a, 2);
    Tensor *b = tensor_ones(shape_b, 2);
    
    assert(network_accuracy(a, b) == 0.0f);
    
    tensor_free(a);
    tensor_free(b);
}

// ============ Training Tests ============

TEST(network_train_single_step) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(2, 1));
    tensor_fill(layer->weights, 0.5f);
    tensor_fill(layer->bias, 0.0f);
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    network_add_layer(net, layer);
    
    OptimizerConfig config = SGD(0.1f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t input_shape[] = {2};
    Tensor *input = tensor_ones(input_shape, 1);
    size_t target_shape[] = {1};
    Tensor *target = tensor_create(target_shape, 1);
    target->data[0] = 2.0f;
    
    float initial_weight = layer->weights->data[0];
    
    float loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    assert(loss >= 0.0f);
    assert(layer->weights->data[0] != initial_weight);
    
    tensor_free(input);
    tensor_free(target);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_multiple_epochs) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(2, 1));
    tensor_fill(layer->weights, 0.5f);
    tensor_fill(layer->bias, 0.0f);
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    network_add_layer(net, layer);
    
    OptimizerConfig config = SGD(0.1f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t input_shape[] = {2};
    Tensor *input = tensor_ones(input_shape, 1);
    size_t target_shape[] = {1};
    Tensor *target = tensor_create(target_shape, 1);
    target->data[0] = 2.0f;
    
    float first_loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    for (int i = 0; i < 10; i++) {
        network_train_step(net, input, target, opt, LOSS_MSE);
    }
    
    float last_loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    assert(last_loss < first_loss);
    
    tensor_free(input);
    tensor_free(target);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_batch) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(3, 2));
    tensor_fill(layer->weights, 0.5f);
    tensor_fill(layer->bias, 0.0f);
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    network_add_layer(net, layer);
    
    OptimizerConfig config = SGD(0.01f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t batch_shape[] = {4, 3};
    Tensor *inputs = tensor_randn(batch_shape, 2, 42);
    size_t target_shape[] = {4, 2};
    Tensor *targets = tensor_randn(target_shape, 2, 123);
    
    float loss = network_train_step(net, inputs, targets, opt, LOSS_MSE);
    
    assert(loss >= 0.0f);
    assert(layer->weights->grad != NULL);
    
    tensor_free(inputs);
    tensor_free(targets);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_with_different_loss_functions) {
    Network *net = network_create();
    
    Layer *layer1 = layer_create(LINEAR(3, 4));
    Layer *layer2 = layer_create(RELU());
    Layer *layer3 = layer_create(LINEAR(4, 2));
    tensor_set_requires_grad(layer1->weights, 1);
    tensor_set_requires_grad(layer1->bias, 1);
    tensor_set_requires_grad(layer3->weights, 1);
    tensor_set_requires_grad(layer3->bias, 1);
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    
    OptimizerConfig config = SGD(0.01f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t input_shape[] = {3};
    Tensor *input = tensor_randn(input_shape, 1, 42);
    size_t target_shape[] = {2};
    Tensor *target = tensor_create(target_shape, 1);
    target->data[0] = 1.0f;
    target->data[1] = 0.0f;
    
    float loss_mse = network_train_step(net, input, target, opt, LOSS_MSE);
    assert(loss_mse >= 0.0f);
    
    optimizer_zero_grad(opt);
    
    float loss_ce = network_train_step(net, input, target, opt, LOSS_CROSS_ENTROPY);
    assert(loss_ce >= 0.0f);
    
    optimizer_zero_grad(opt);
    
    float loss_bce = network_train_step(net, input, target, opt, LOSS_BINARY_CROSS_ENTROPY);
    assert(loss_bce >= 0.0f);
    
    tensor_free(input);
    tensor_free(target);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_convergence) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(1, 1));
    tensor_fill(layer->weights, 0.1f);
    tensor_fill(layer->bias, 0.1f);
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    network_add_layer(net, layer);
    
    OptimizerConfig config = SGD(0.1f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t shape[] = {1};
    Tensor *input = tensor_create(shape, 1);
    input->data[0] = 2.0f;
    Tensor *target = tensor_create(shape, 1);
    target->data[0] = 5.0f;
    
    float initial_loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    for (int i = 0; i < 100; i++) {
        network_train_step(net, input, target, opt, LOSS_MSE);
    }
    
    float final_loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    assert(final_loss < initial_loss);
    assert(final_loss < 0.1f);
    
    tensor_free(input);
    tensor_free(target);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_with_momentum) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(2, 1));
    tensor_fill(layer->weights, 0.5f);
    tensor_fill(layer->bias, 0.0f);
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    network_add_layer(net, layer);
    
    OptimizerConfig config = SGD(0.1f, 0.9f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t input_shape[] = {2};
    Tensor *input = tensor_ones(input_shape, 1);
    size_t target_shape[] = {1};
    Tensor *target = tensor_create(target_shape, 1);
    target->data[0] = 2.0f;
    
    float first_loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    for (int i = 0; i < 20; i++) {
        network_train_step(net, input, target, opt, LOSS_MSE);
    }
    
    float last_loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    assert(last_loss < first_loss);
    
    tensor_free(input);
    tensor_free(target);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_with_adam) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(2, 1));
    tensor_fill(layer->weights, 0.5f);
    tensor_fill(layer->bias, 0.0f);
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    network_add_layer(net, layer);
    
    OptimizerConfig config = ADAM(0.01f, 0.9f, 0.999f, 1e-8f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t input_shape[] = {2};
    Tensor *input = tensor_ones(input_shape, 1);
    size_t target_shape[] = {1};
    Tensor *target = tensor_create(target_shape, 1);
    target->data[0] = 2.0f;
    
    float first_loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    for (int i = 0; i < 50; i++) {
        network_train_step(net, input, target, opt, LOSS_MSE);
    }
    
    float last_loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    assert(last_loss < first_loss);
    
    tensor_free(input);
    tensor_free(target);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_classification) {
    Network *net = network_create();
    
    Layer *layer1 = layer_create(LINEAR(2, 2));
    tensor_set_requires_grad(layer1->weights, 1);
    tensor_set_requires_grad(layer1->bias, 1);
    
    network_add_layer(net, layer1);
    
    OptimizerConfig config = SGD(0.1f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t input_shape[] = {2};
    Tensor *input = tensor_create(input_shape, 1);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    
    size_t target_shape[] = {2};
    Tensor *target = tensor_create(target_shape, 1);
    target->data[0] = 3.0f;
    target->data[1] = 1.0f;
    
    float first_loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    for (int i = 0; i < 20; i++) {
        network_train_step(net, input, target, opt, LOSS_MSE);
    }
    
    float last_loss = network_train_step(net, input, target, opt, LOSS_MSE);
    
    assert(last_loss < first_loss);
    
    tensor_free(input);
    tensor_free(target);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_null_checks) {
    Network *net = network_create();
    Layer *layer = layer_create(LINEAR(2, 1));
    network_add_layer(net, layer);
    
    OptimizerConfig config = SGD(0.1f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t shape[] = {2};
    Tensor *input = tensor_ones(shape, 1);
    Tensor *target = tensor_ones(shape, 1);
    
    assert(network_train_step(NULL, input, target, opt, LOSS_MSE) == 0.0f);
    assert(network_train_step(net, NULL, target, opt, LOSS_MSE) == 0.0f);
    assert(network_train_step(net, input, NULL, opt, LOSS_MSE) == 0.0f);
    assert(network_train_step(net, input, target, NULL, LOSS_MSE) == 0.0f);
    
    tensor_free(input);
    tensor_free(target);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_complex_classification) {
    Network *net = network_create();
    
    Layer *layer1 = layer_create(LINEAR(4, 8));
    Layer *layer2 = layer_create(TANH());
    Layer *layer3 = layer_create(LINEAR(8, 6));
    Layer *layer4 = layer_create(TANH());
    Layer *layer5 = layer_create(LINEAR(6, 3));
    Layer *layer6 = layer_create(SIGMOID());
    
    tensor_set_requires_grad(layer1->weights, 1);
    tensor_set_requires_grad(layer1->bias, 1);
    tensor_set_requires_grad(layer3->weights, 1);
    tensor_set_requires_grad(layer3->bias, 1);
    tensor_set_requires_grad(layer5->weights, 1);
    tensor_set_requires_grad(layer5->bias, 1);
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    network_add_layer(net, layer4);
    network_add_layer(net, layer5);
    network_add_layer(net, layer6);
    
    OptimizerConfig config = ADAM(0.01f, 0.9f, 0.999f, 1e-8f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t input_shape[] = {4};
    Tensor *inputs = tensor_create(input_shape, 1);
    
    inputs->data[0] = 1.0f;
    inputs->data[1] = 0.0f;
    inputs->data[2] = 1.0f;
    inputs->data[3] = 0.0f;
    
    size_t target_shape[] = {3};
    Tensor *targets = tensor_create(target_shape, 1);
    
    targets->data[0] = 1.0f;
    targets->data[1] = 0.0f;
    targets->data[2] = 0.0f;
    
    float initial_loss = network_train_step(net, inputs, targets, opt, LOSS_MSE);
    assert(initial_loss > 0.0f);
    
    for (int epoch = 0; epoch < 200; epoch++) {
        network_train_step(net, inputs, targets, opt, LOSS_MSE);
    }
    
    float final_loss = network_train_step(net, inputs, targets, opt, LOSS_MSE);
    
    assert(final_loss < initial_loss);
    assert(final_loss < initial_loss * 0.5f);
    
    Tensor *predictions = network_forward(net, inputs);
    assert(predictions != NULL);
    assert(predictions->data[0] > 0.3f);
    
    tensor_free(inputs);
    tensor_free(targets);
    tensor_free(predictions);
    optimizer_free(opt);
    network_free(net);
}

TEST(network_train_very_deep_multiclass_batch) {
    Network *net = network_create();
    
    Layer *layer1 = layer_create(LINEAR(5, 16));
    Layer *layer2 = layer_create(RELU());
    Layer *layer3 = layer_create(LINEAR(16, 12));
    Layer *layer4 = layer_create(TANH());
    Layer *layer5 = layer_create(LINEAR(12, 10));
    Layer *layer6 = layer_create(RELU());
    Layer *layer7 = layer_create(LINEAR(10, 8));
    Layer *layer8 = layer_create(TANH());
    Layer *layer9 = layer_create(LINEAR(8, 4));
    Layer *layer10 = layer_create(SIGMOID());
    
    tensor_set_requires_grad(layer1->weights, 1);
    tensor_set_requires_grad(layer1->bias, 1);
    tensor_set_requires_grad(layer3->weights, 1);
    tensor_set_requires_grad(layer3->bias, 1);
    tensor_set_requires_grad(layer5->weights, 1);
    tensor_set_requires_grad(layer5->bias, 1);
    tensor_set_requires_grad(layer7->weights, 1);
    tensor_set_requires_grad(layer7->bias, 1);
    tensor_set_requires_grad(layer9->weights, 1);
    tensor_set_requires_grad(layer9->bias, 1);
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    network_add_layer(net, layer4);
    network_add_layer(net, layer5);
    network_add_layer(net, layer6);
    network_add_layer(net, layer7);
    network_add_layer(net, layer8);
    network_add_layer(net, layer9);
    network_add_layer(net, layer10);
    
    OptimizerConfig config = ADAM(0.005f, 0.9f, 0.999f, 1e-8f);
    Optimizer *opt = optimizer_create(net, config);
    
    size_t input_shape[] = {12, 5};
    Tensor *inputs = tensor_create(input_shape, 2);
    
    inputs->data[0] = 1.0f; inputs->data[1] = 0.0f; inputs->data[2] = 1.0f; inputs->data[3] = 0.0f; inputs->data[4] = 1.0f;
    inputs->data[5] = 0.0f; inputs->data[6] = 1.0f; inputs->data[7] = 0.0f; inputs->data[8] = 1.0f; inputs->data[9] = 0.0f;
    inputs->data[10] = 1.0f; inputs->data[11] = 1.0f; inputs->data[12] = 0.0f; inputs->data[13] = 0.0f; inputs->data[14] = 1.0f;
    inputs->data[15] = 0.0f; inputs->data[16] = 0.0f; inputs->data[17] = 1.0f; inputs->data[18] = 1.0f; inputs->data[19] = 0.0f;
    inputs->data[20] = 1.0f; inputs->data[21] = 0.0f; inputs->data[22] = 0.0f; inputs->data[23] = 1.0f; inputs->data[24] = 1.0f;
    inputs->data[25] = 0.0f; inputs->data[26] = 1.0f; inputs->data[27] = 1.0f; inputs->data[28] = 0.0f; inputs->data[29] = 0.0f;
    inputs->data[30] = 1.0f; inputs->data[31] = 1.0f; inputs->data[32] = 1.0f; inputs->data[33] = 0.0f; inputs->data[34] = 0.0f;
    inputs->data[35] = 0.0f; inputs->data[36] = 0.0f; inputs->data[37] = 0.0f; inputs->data[38] = 1.0f; inputs->data[39] = 1.0f;
    inputs->data[40] = 1.0f; inputs->data[41] = 0.5f; inputs->data[42] = 1.0f; inputs->data[43] = 0.5f; inputs->data[44] = 0.0f;
    inputs->data[45] = 0.5f; inputs->data[46] = 1.0f; inputs->data[47] = 0.0f; inputs->data[48] = 1.0f; inputs->data[49] = 0.5f;
    inputs->data[50] = 0.0f; inputs->data[51] = 0.5f; inputs->data[52] = 0.5f; inputs->data[53] = 1.0f; inputs->data[54] = 1.0f;
    inputs->data[55] = 1.0f; inputs->data[56] = 1.0f; inputs->data[57] = 0.5f; inputs->data[58] = 0.5f; inputs->data[59] = 0.0f;
    
    size_t target_shape[] = {12, 4};
    Tensor *targets = tensor_create(target_shape, 2);
    
    targets->data[0] = 1.0f; targets->data[1] = 0.0f; targets->data[2] = 0.0f; targets->data[3] = 0.0f;
    targets->data[4] = 0.0f; targets->data[5] = 1.0f; targets->data[6] = 0.0f; targets->data[7] = 0.0f;
    targets->data[8] = 1.0f; targets->data[9] = 0.0f; targets->data[10] = 0.0f; targets->data[11] = 0.0f;
    targets->data[12] = 0.0f; targets->data[13] = 0.0f; targets->data[14] = 1.0f; targets->data[15] = 0.0f;
    targets->data[16] = 1.0f; targets->data[17] = 0.0f; targets->data[18] = 0.0f; targets->data[19] = 0.0f;
    targets->data[20] = 0.0f; targets->data[21] = 1.0f; targets->data[22] = 0.0f; targets->data[23] = 0.0f;
    targets->data[24] = 1.0f; targets->data[25] = 0.0f; targets->data[26] = 0.0f; targets->data[27] = 0.0f;
    targets->data[28] = 0.0f; targets->data[29] = 0.0f; targets->data[30] = 0.0f; targets->data[31] = 1.0f;
    targets->data[32] = 0.0f; targets->data[33] = 1.0f; targets->data[34] = 0.0f; targets->data[35] = 0.0f;
    targets->data[36] = 0.0f; targets->data[37] = 0.0f; targets->data[38] = 1.0f; targets->data[39] = 0.0f;
    targets->data[40] = 0.0f; targets->data[41] = 0.0f; targets->data[42] = 0.0f; targets->data[43] = 1.0f;
    targets->data[44] = 1.0f; targets->data[45] = 0.0f; targets->data[46] = 0.0f; targets->data[47] = 0.0f;
    
    printf("  Training 10-layer network on 12 samples with 4 classes...\n");
    
    float initial_loss = network_train_step(net, inputs, targets, opt, LOSS_MSE);
    assert(initial_loss > 0.0f);
    printf("  Initial loss: %.6f\n", initial_loss);
    
    float weight_before = layer1->weights->data[0];
    
    for (int epoch = 0; epoch < 300; epoch++) {
        float loss = network_train_step(net, inputs, targets, opt, LOSS_MSE);
        if (epoch % 100 == 0) {
            printf("  Epoch %d: loss = %.6f\n", epoch, loss);
        }
    }
    
    float final_loss = network_train_step(net, inputs, targets, opt, LOSS_MSE);
    printf("  Final loss: %.6f\n", final_loss);
    
    float weight_after = layer1->weights->data[0];
    printf("  Weight changed: %.6f -> %.6f (delta: %.6f)\n", 
           weight_before, weight_after, weight_after - weight_before);
    
    assert(final_loss < initial_loss);
    assert(final_loss < initial_loss * 0.3f);
    
    Tensor *predictions = network_forward(net, inputs);
    assert(predictions != NULL);
    assert(predictions->ndim == 2);
    assert(predictions->shape[0] == 12);
    assert(predictions->shape[1] == 4);
    
    float acc = network_accuracy(predictions, targets);
    printf("  Classification accuracy: %.2f%%\n", acc * 100.0f);
    assert(acc >= 0.4f);
    
    tensor_free(inputs);
    tensor_free(targets);
    tensor_free(predictions);
    optimizer_free(opt);
    network_free(net);
}

// ============ Complex Network Tests ============

TEST(network_deep_network) {
    Network *net = network_create();
    
    // Create a deep network with 5 layers
    network_add_layer(net, layer_create(LINEAR(10, 8)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(8, 6)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(6, 4)));
    
    assert(net->num_layers == 5);
    
    size_t input_shape[] = {10};
    Tensor *input = tensor_randn(input_shape, 1, 42);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 4);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_with_all_activation_types) {
    Network *net = network_create();
    
    network_add_layer(net, layer_create(LINEAR(5, 5)));
    network_add_layer(net, layer_create(RELU()));
    network_add_layer(net, layer_create(LINEAR(5, 5)));
    network_add_layer(net, layer_create(SIGMOID()));
    network_add_layer(net, layer_create(LINEAR(5, 5)));
    network_add_layer(net, layer_create(TANH()));
    network_add_layer(net, layer_create(LINEAR(5, 3)));
    network_add_layer(net, layer_create(SOFTMAX()));
    
    size_t input_shape[] = {5};
    Tensor *input = tensor_randn(input_shape, 1, 42);
    
    Tensor *output = network_forward(net, input);
    
    assert(output != NULL);
    assert(output->shape[0] == 3);
    
    // Softmax output should sum to 1
    float sum = 0.0f;
    for (size_t i = 0; i < output->size; i++) {
        sum += output->data[i];
    }
    ASSERT_FLOAT_EQ(sum, 1.0f);
    
    tensor_free(input);
    tensor_free(output);
    network_free(net);
}

TEST(network_batch_processing) {
    Network *net = network_create();
    
    Layer *layer = layer_create(LINEAR(4, 2));
    tensor_fill(layer->weights, 1.0f);
    tensor_fill(layer->bias, 0.0f);
    network_add_layer(net, layer);
    
    // Process batches of different sizes
    size_t batch_sizes[] = {1, 5, 10, 32};
    
    for (size_t i = 0; i < 4; i++) {
        size_t input_shape[] = {batch_sizes[i], 4};
        Tensor *input = tensor_ones(input_shape, 2);
        
        Tensor *output = network_forward(net, input);
        
        assert(output != NULL);
        assert(output->shape[0] == batch_sizes[i]);
        assert(output->shape[1] == 2);
        
        tensor_free(input);
        tensor_free(output);
    }
    
    network_free(net);
}

int main() {
    printf("\n=== Running Network Unit Tests ===\n\n");
    
    // Network creation
    RUN_TEST(network_create_basic);
    RUN_TEST(network_add_single_layer);
    RUN_TEST(network_add_multiple_layers);
    RUN_TEST(network_capacity_expansion);
    
    // Forward pass
    RUN_TEST(network_forward_single_layer);
    RUN_TEST(network_forward_multi_layer);
    RUN_TEST(network_forward_batch);
    RUN_TEST(network_forward_with_activations);
    
    // Parameter management
    RUN_TEST(network_get_parameters_empty);
    RUN_TEST(network_get_parameters_single_layer);
    RUN_TEST(network_get_parameters_multiple_layers);
    RUN_TEST(network_zero_grad);
    
    // Accuracy
    RUN_TEST(network_accuracy_perfect);
    RUN_TEST(network_accuracy_partial);
    RUN_TEST(network_accuracy_zero);
    
    // Training
    RUN_TEST(network_train_single_step);
    RUN_TEST(network_train_multiple_epochs);
    RUN_TEST(network_train_batch);
    RUN_TEST(network_train_with_different_loss_functions);
    RUN_TEST(network_train_convergence);
    RUN_TEST(network_train_with_momentum);
    RUN_TEST(network_train_with_adam);
    RUN_TEST(network_train_classification);
    RUN_TEST(network_train_null_checks);
    RUN_TEST(network_train_complex_classification);
    RUN_TEST(network_train_very_deep_multiclass_batch);
    
    // Edge cases
    RUN_TEST(network_forward_null_input);
    RUN_TEST(network_forward_null_network);
    RUN_TEST(network_forward_empty_network);
    RUN_TEST(network_free_null);
    RUN_TEST(network_add_layer_null_network);
    RUN_TEST(network_add_layer_null_layer);
    RUN_TEST(network_zero_grad_null);
    RUN_TEST(network_accuracy_null_inputs);
    RUN_TEST(network_accuracy_mismatched_shapes);
    
    // Complex networks
    RUN_TEST(network_deep_network);
    RUN_TEST(network_with_all_activation_types);
    RUN_TEST(network_batch_processing);
    
    printf("\n=== All Network Tests Passed! ===\n\n");
    return 0;
}
