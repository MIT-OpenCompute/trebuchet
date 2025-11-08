/*
 * AI Generated Unit tests for optimizer.c
 * Tests SGD and Adam optimizers, parameter updates, and state management
 */

#include "lessbasednn/optimizer.h"
#include "lessbasednn/network.h"
#include "lessbasednn/layer.h"
#include "lessbasednn/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define EPSILON 1e-5
#define ASSERT_FLOAT_EQ(a, b) assert(fabsf((a) - (b)) < EPSILON)
#define ASSERT_FLOAT_NEAR(a, b, eps) assert(fabsf((a) - (b)) < (eps))
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    printf("Running test_%s...\n", #name); \
    test_##name(); \
    printf("  ✓ test_%s passed\n", #name); \
} while(0)

// Helper function to create a simple network
static Network* create_test_network() {
    Network *net = network_create();
    Layer *layer = layer_create(LINEAR(3, 2));
    tensor_fill(layer->weights, 1.0f);
    tensor_fill(layer->bias, 0.5f);
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    network_add_layer(net, layer);
    return net;
}

// ============ SGD Optimizer Tests ============

TEST(optimizer_sgd_create_basic) {
    Network *net = create_test_network();
    
    OptimizerConfig config = SGD(0.01f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    assert(opt != NULL);
    assert(opt->type == OPTIMIZER_SGD);
    assert(opt->num_parameters == 2);
    assert(opt->parameters != NULL);
    assert(opt->step != NULL);
    assert(opt->zero_grad != NULL);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_sgd_create_with_momentum) {
    Network *net = create_test_network();
    
    OptimizerConfig config = SGD(0.01f, 0.9f);
    Optimizer *opt = optimizer_create(net, config);
    
    assert(opt != NULL);
    assert(opt->type == OPTIMIZER_SGD);
    assert(opt->state != NULL);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_sgd_step_basic) {
    Network *net = create_test_network();
    
    OptimizerConfig config = SGD(0.1f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    // Get parameters
    Layer *layer = net->layers[0];
    
    // Set up gradients
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    layer->bias->grad = (float*)calloc(layer->bias->size, sizeof(float));
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 1.0f;
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        layer->bias->grad[i] = 1.0f;
    }
    
    // Store original values
    float original_weight = layer->weights->data[0];
    float original_bias = layer->bias->data[0];
    
    // Perform step
    optimizer_step(opt);
    
    // Parameters should have decreased by learning_rate * gradient
    // new_value = old_value - lr * grad = 1.0 - 0.1 * 1.0 = 0.9
    ASSERT_FLOAT_EQ(layer->weights->data[0], 0.9f);
    ASSERT_FLOAT_EQ(layer->bias->data[0], 0.4f);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_sgd_step_with_momentum) {
    Network *net = create_test_network();
    
    OptimizerConfig config = SGD(0.1f, 0.9f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    
    // Allocate gradients
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 1.0f;
    }
    
    float original = layer->weights->data[0];
    
    // First step
    optimizer_step(opt);
    float after_first = layer->weights->data[0];
    
    // Second step with same gradient
    optimizer_step(opt);
    float after_second = layer->weights->data[0];
    
    // With momentum, second step should have larger change
    float first_change = fabsf(after_first - original);
    float second_change = fabsf(after_second - after_first);
    assert(second_change > first_change);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_sgd_multiple_steps) {
    Network *net = create_test_network();
    
    OptimizerConfig config = SGD(0.01f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 0.5f;
    }
    
    float initial = layer->weights->data[0];
    
    // Perform 10 optimization steps
    for (int i = 0; i < 10; i++) {
        optimizer_step(opt);
    }
    
    // Should have moved by 10 * 0.01 * 0.5 = 0.05
    ASSERT_FLOAT_NEAR(layer->weights->data[0], initial - 0.05f, 1e-4);
    
    optimizer_free(opt);
    network_free(net);
}

// ============ Adam Optimizer Tests ============

TEST(optimizer_adam_create_basic) {
    Network *net = create_test_network();
    
    OptimizerConfig config = ADAM(0.001f, 0.9f, 0.999f, 1e-8f);
    Optimizer *opt = optimizer_create(net, config);
    
    assert(opt != NULL);
    assert(opt->type == OPTIMIZER_ADAM);
    assert(opt->num_parameters == 2);
    assert(opt->state != NULL);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_adam_step_basic) {
    Network *net = create_test_network();
    
    OptimizerConfig config = ADAM(0.001f, 0.9f, 0.999f, 1e-8f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    
    // Set up gradients
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    layer->bias->grad = (float*)calloc(layer->bias->size, sizeof(float));
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 1.0f;
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        layer->bias->grad[i] = 1.0f;
    }
    
    float original = layer->weights->data[0];
    
    optimizer_step(opt);
    
    // Parameters should have changed
    assert(layer->weights->data[0] != original);
    assert(layer->weights->data[0] < original);  // Should decrease with positive gradient
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_adam_bias_correction) {
    Network *net = create_test_network();
    
    OptimizerConfig config = ADAM(0.001f, 0.9f, 0.999f, 1e-8f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 1.0f;
    }
    
    // First step should have bias correction
    float before_first = layer->weights->data[0];
    optimizer_step(opt);
    float after_first = layer->weights->data[0];
    float first_change = fabsf(after_first - before_first);
    
    // Many steps later, bias correction is less significant
    for (int i = 0; i < 99; i++) {
        optimizer_step(opt);
    }
    
    float before_100th = layer->weights->data[0];
    optimizer_step(opt);
    float after_100th = layer->weights->data[0];
    float later_change = fabsf(after_100th - before_100th);
    
    // With constant gradients, Adam should give consistent step sizes
    // Bias correction prevents initial steps from being too small
    assert(fabsf(first_change - later_change) < 0.0001f);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_adam_adaptive_learning_rate) {
    Network *net = create_test_network();
    
    OptimizerConfig config = ADAM(0.001f, 0.9f, 0.999f, 1e-8f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    
    // Set different gradients for different parameters
    layer->weights->grad[0] = 0.1f;  // Small gradient
    layer->weights->grad[1] = 10.0f; // Large gradient
    
    float param0_before = layer->weights->data[0];
    float param1_before = layer->weights->data[1];
    
    optimizer_step(opt);
    
    float param0_change = fabsf(layer->weights->data[0] - param0_before);
    float param1_change = fabsf(layer->weights->data[1] - param1_before);
    
    // Adam should adapt: larger gradients don't necessarily mean larger steps
    // This is the adaptive part of Adam
    assert(param0_change > 0);
    assert(param1_change > 0);
    
    optimizer_free(opt);
    network_free(net);
}

// ============ Optimizer Zero Grad Tests ============

TEST(optimizer_zero_grad_sgd) {
    Network *net = create_test_network();
    
    OptimizerConfig config = SGD(0.01f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    
    // Set up gradients
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    layer->bias->grad = (float*)calloc(layer->bias->size, sizeof(float));
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 5.0f;
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        layer->bias->grad[i] = 3.0f;
    }
    
    optimizer_zero_grad(opt);
    
    // All gradients should be zero
    for (size_t i = 0; i < layer->weights->size; i++) {
        ASSERT_FLOAT_EQ(layer->weights->grad[i], 0.0f);
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        ASSERT_FLOAT_EQ(layer->bias->grad[i], 0.0f);
    }
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_zero_grad_adam) {
    Network *net = create_test_network();
    
    OptimizerConfig config = ADAM(0.001f, 0.9f, 0.999f, 1e-8f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 2.5f;
    }
    
    optimizer_zero_grad(opt);
    
    for (size_t i = 0; i < layer->weights->size; i++) {
        ASSERT_FLOAT_EQ(layer->weights->grad[i], 0.0f);
    }
    
    optimizer_free(opt);
    network_free(net);
}

// ============ Edge Cases and Error Handling ============

TEST(optimizer_create_null_network) {
    OptimizerConfig config = SGD(0.01f, 0.0f);
    Optimizer *opt = optimizer_create(NULL, config);
    
    assert(opt == NULL);
}

TEST(optimizer_step_null) {
    optimizer_step(NULL);  // Should not crash
}

TEST(optimizer_zero_grad_null) {
    optimizer_zero_grad(NULL);  // Should not crash
}

TEST(optimizer_free_null) {
    optimizer_free(NULL);  // Should not crash
}

TEST(optimizer_sgd_no_gradient) {
    Network *net = create_test_network();
    
    OptimizerConfig config = SGD(0.01f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    float original = layer->weights->data[0];
    
    // Don't allocate gradients
    
    optimizer_step(opt);
    
    // Parameters should not change without gradients
    ASSERT_FLOAT_EQ(layer->weights->data[0], original);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_adam_no_gradient) {
    Network *net = create_test_network();
    
    OptimizerConfig config = ADAM(0.001f, 0.9f, 0.999f, 1e-8f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    float original = layer->weights->data[0];
    
    // Don't allocate gradients
    
    optimizer_step(opt);
    
    // Parameters should not change without gradients
    ASSERT_FLOAT_EQ(layer->weights->data[0], original);
    
    optimizer_free(opt);
    network_free(net);
}

// ============ Different Learning Rate Tests ============

TEST(optimizer_different_learning_rates) {
    float learning_rates[] = {0.001f, 0.01f, 0.1f, 1.0f};
    
    for (size_t lr_idx = 0; lr_idx < 4; lr_idx++) {
        Network *net = create_test_network();
        
        OptimizerConfig config = SGD(learning_rates[lr_idx], 0.0f);
        Optimizer *opt = optimizer_create(net, config);
        
        Layer *layer = net->layers[0];
        
        layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
        layer->weights->grad[0] = 1.0f;
        
        float before = layer->weights->data[0];
        optimizer_step(opt);
        float after = layer->weights->data[0];
        
        float expected_change = learning_rates[lr_idx] * 1.0f;
        ASSERT_FLOAT_NEAR(before - after, expected_change, 1e-5);
        
        optimizer_free(opt);
        network_free(net);
    }
}

// ============ Multiple Parameter Tests ============

TEST(optimizer_multiple_parameters) {
    Network *net = network_create();
    
    // Add multiple layers
    Layer *layer1 = layer_create(LINEAR(5, 4));
    Layer *layer2 = layer_create(LINEAR(4, 3));
    Layer *layer3 = layer_create(LINEAR(3, 2));
    
    tensor_set_requires_grad(layer1->weights, 1);
    tensor_set_requires_grad(layer1->bias, 1);
    tensor_set_requires_grad(layer2->weights, 1);
    tensor_set_requires_grad(layer2->bias, 1);
    tensor_set_requires_grad(layer3->weights, 1);
    tensor_set_requires_grad(layer3->bias, 1);
    
    network_add_layer(net, layer1);
    network_add_layer(net, layer2);
    network_add_layer(net, layer3);
    
    OptimizerConfig config = SGD(0.01f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    assert(opt->num_parameters == 6);  // 3 weights + 3 biases
    
    // Allocate gradients for all parameters
    for (size_t i = 0; i < opt->num_parameters; i++) {
        opt->parameters[i]->grad = (float*)calloc(opt->parameters[i]->size, sizeof(float));
        for (size_t j = 0; j < opt->parameters[i]->size; j++) {
            opt->parameters[i]->grad[j] = 1.0f;
        }
    }
    
    // Store original values
    float originals[6];
    for (size_t i = 0; i < 6; i++) {
        originals[i] = opt->parameters[i]->data[0];
    }
    
    optimizer_step(opt);
    
    // All parameters should have been updated
    for (size_t i = 0; i < 6; i++) {
        assert(opt->parameters[i]->data[0] != originals[i]);
    }
    
    optimizer_free(opt);
    network_free(net);
}

// ============ Training Simulation Tests ============

TEST(optimizer_training_loop_simulation) {
    Network *net = create_test_network();
    
    OptimizerConfig config = SGD(0.1f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    
    // Simulate a training loop
    for (int epoch = 0; epoch < 5; epoch++) {
        // Simulate gradient computation
        layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
        layer->bias->grad = (float*)calloc(layer->bias->size, sizeof(float));
        
        for (size_t i = 0; i < layer->weights->size; i++) {
            layer->weights->grad[i] = 0.5f;
        }
        for (size_t i = 0; i < layer->bias->size; i++) {
            layer->bias->grad[i] = 0.5f;
        }
        
        // Optimizer step
        optimizer_step(opt);
        
        // Zero gradients
        optimizer_zero_grad(opt);
    }
    
    // After 5 epochs, parameters should have moved significantly
    assert(fabsf(layer->weights->data[0] - 1.0f) > 0.1f);
    
    optimizer_free(opt);
    network_free(net);
}

TEST(optimizer_convergence_test) {
    Network *net = create_test_network();
    
    OptimizerConfig config = SGD(0.5f, 0.0f);
    Optimizer *opt = optimizer_create(net, config);
    
    Layer *layer = net->layers[0];
    
    // Simulate converging to zero
    for (int i = 0; i < 100; i++) {
        if (!layer->weights->grad) {
            layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
        }
        
        // Gradient points towards zero
        for (size_t j = 0; j < layer->weights->size; j++) {
            layer->weights->grad[j] = layer->weights->data[j];
        }
        
        optimizer_step(opt);
    }
    
    // Should be close to zero
    for (size_t i = 0; i < layer->weights->size; i++) {
        assert(fabsf(layer->weights->data[i]) < 0.01f);
    }
    
    optimizer_free(opt);
    network_free(net);
}

int main() {
    printf("\n=== Running Optimizer Unit Tests ===\n\n");
    
    // SGD tests
    RUN_TEST(optimizer_sgd_create_basic);
    RUN_TEST(optimizer_sgd_create_with_momentum);
    RUN_TEST(optimizer_sgd_step_basic);
    RUN_TEST(optimizer_sgd_step_with_momentum);
    RUN_TEST(optimizer_sgd_multiple_steps);
    
    // Adam tests
    RUN_TEST(optimizer_adam_create_basic);
    RUN_TEST(optimizer_adam_step_basic);
    RUN_TEST(optimizer_adam_bias_correction);
    RUN_TEST(optimizer_adam_adaptive_learning_rate);
    
    // Zero grad tests
    RUN_TEST(optimizer_zero_grad_sgd);
    RUN_TEST(optimizer_zero_grad_adam);
    
    // Edge cases
    RUN_TEST(optimizer_create_null_network);
    RUN_TEST(optimizer_step_null);
    RUN_TEST(optimizer_zero_grad_null);
    RUN_TEST(optimizer_free_null);
    RUN_TEST(optimizer_sgd_no_gradient);
    RUN_TEST(optimizer_adam_no_gradient);
    
    // Learning rate tests
    RUN_TEST(optimizer_different_learning_rates);
    
    // Multiple parameters
    RUN_TEST(optimizer_multiple_parameters);
    
    // Training simulation
    RUN_TEST(optimizer_training_loop_simulation);
    RUN_TEST(optimizer_convergence_test);
    
    printf("\n=== All Optimizer Tests Passed! ===\n\n");
    return 0;
}
