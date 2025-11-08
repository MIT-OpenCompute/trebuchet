/*
 * AI generated Unit tests for layer.c
 * Tests layer creation, forward pass, parameter management, and different layer types
 */

#include "lessbasednn/layer.h"
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

// ============ Layer Creation Tests ============

TEST(layer_create_linear) {
    LayerConfig config = LINEAR(10, 5);
    Layer *layer = layer_create(config);
    
    assert(layer != NULL);
    assert(layer->type == LAYER_LINEAR);
    assert(layer->weights != NULL);
    assert(layer->bias != NULL);
    assert(layer->num_parameters == 2);
    assert(layer->parameters != NULL);
    
    // Check weights shape
    assert(layer->weights->ndim == 2);
    assert(layer->weights->shape[0] == 10);
    assert(layer->weights->shape[1] == 5);
    
    // Check bias shape
    assert(layer->bias->ndim == 1);
    assert(layer->bias->shape[0] == 5);
    
    // Check bias is zeros
    for (size_t i = 0; i < layer->bias->size; i++) {
        ASSERT_FLOAT_EQ(layer->bias->data[i], 0.0f);
    }
    
    layer_free(layer);
}

TEST(layer_create_relu) {
    LayerConfig config = RELU();
    Layer *layer = layer_create(config);
    
    assert(layer != NULL);
    assert(layer->type == LAYER_RELU);
    assert(layer->weights == NULL);
    assert(layer->bias == NULL);
    assert(layer->num_parameters == 0);
    assert(layer->forward != NULL);
    
    layer_free(layer);
}

TEST(layer_create_sigmoid) {
    LayerConfig config = SIGMOID();
    Layer *layer = layer_create(config);
    
    assert(layer != NULL);
    assert(layer->type == LAYER_SIGMOID);
    assert(layer->weights == NULL);
    assert(layer->bias == NULL);
    assert(layer->num_parameters == 0);
    
    layer_free(layer);
}

TEST(layer_create_tanh) {
    LayerConfig config = TANH();
    Layer *layer = layer_create(config);
    
    assert(layer != NULL);
    assert(layer->type == LAYER_TANH);
    assert(layer->weights == NULL);
    assert(layer->bias == NULL);
    assert(layer->num_parameters == 0);
    
    layer_free(layer);
}

TEST(layer_create_softmax) {
    LayerConfig config = SOFTMAX();
    Layer *layer = layer_create(config);
    
    assert(layer != NULL);
    assert(layer->type == LAYER_SOFTMAX);
    assert(layer->weights == NULL);
    assert(layer->bias == NULL);
    assert(layer->num_parameters == 0);
    
    layer_free(layer);
}

// ============ Layer Forward Tests ============

TEST(layer_forward_linear_single) {
    LayerConfig config = LINEAR(3, 2);
    Layer *layer = layer_create(config);
    
    // Set known weights and bias
    tensor_fill(layer->weights, 1.0f);
    tensor_fill(layer->bias, 0.5f);
    
    // Create input
    size_t input_shape[] = {3};
    Tensor *input = tensor_ones(input_shape, 1);
    
    // Forward pass
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    assert(output->ndim == 1);
    assert(output->shape[0] == 2);
    
    // output = input @ weights + bias
    // [1, 1, 1] @ [[1, 1], [1, 1], [1, 1]] + [0.5, 0.5] = [3.5, 3.5]
    ASSERT_FLOAT_EQ(output->data[0], 3.5f);
    ASSERT_FLOAT_EQ(output->data[1], 3.5f);
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

TEST(layer_forward_linear_batch) {
    LayerConfig config = LINEAR(4, 3);
    Layer *layer = layer_create(config);
    
    tensor_fill(layer->weights, 0.5f);
    tensor_fill(layer->bias, 0.1f);
    
    // Batch input (2 samples, 4 features each)
    size_t input_shape[] = {2, 4};
    Tensor *input = tensor_ones(input_shape, 2);
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    assert(output->ndim == 2);
    assert(output->shape[0] == 2);
    assert(output->shape[1] == 3);
    
    // Each output: [1, 1, 1, 1] @ weights (all 0.5) + bias (0.1)
    // = [2.0, 2.0, 2.0] + [0.1, 0.1, 0.1] = [2.1, 2.1, 2.1]
    for (size_t i = 0; i < output->size; i++) {
        ASSERT_FLOAT_EQ(output->data[i], 2.1f);
    }
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

TEST(layer_forward_relu) {
    LayerConfig config = RELU();
    Layer *layer = layer_create(config);
    
    size_t shape[] = {5};
    Tensor *input = tensor_create(shape, 1);
    input->data[0] = -2.0f;
    input->data[1] = -0.5f;
    input->data[2] = 0.0f;
    input->data[3] = 1.5f;
    input->data[4] = 3.0f;
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    ASSERT_FLOAT_EQ(output->data[0], 0.0f);
    ASSERT_FLOAT_EQ(output->data[1], 0.0f);
    ASSERT_FLOAT_EQ(output->data[2], 0.0f);
    ASSERT_FLOAT_EQ(output->data[3], 1.5f);
    ASSERT_FLOAT_EQ(output->data[4], 3.0f);
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

TEST(layer_forward_sigmoid) {
    LayerConfig config = SIGMOID();
    Layer *layer = layer_create(config);
    
    size_t shape[] = {3};
    Tensor *input = tensor_create(shape, 1);
    input->data[0] = -10.0f;
    input->data[1] = 0.0f;
    input->data[2] = 10.0f;
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    assert(output->data[0] < 0.01f);  // Close to 0
    ASSERT_FLOAT_EQ(output->data[1], 0.5f);
    assert(output->data[2] > 0.99f);  // Close to 1
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

TEST(layer_forward_tanh) {
    LayerConfig config = TANH();
    Layer *layer = layer_create(config);
    
    size_t shape[] = {3};
    Tensor *input = tensor_create(shape, 1);
    input->data[0] = -1.0f;
    input->data[1] = 0.0f;
    input->data[2] = 1.0f;
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    ASSERT_FLOAT_EQ(output->data[0], tanhf(-1.0f));
    ASSERT_FLOAT_EQ(output->data[1], 0.0f);
    ASSERT_FLOAT_EQ(output->data[2], tanhf(1.0f));
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

TEST(layer_forward_softmax) {
    LayerConfig config = SOFTMAX();
    Layer *layer = layer_create(config);
    
    size_t shape[] = {4};
    Tensor *input = tensor_create(shape, 1);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;
    input->data[3] = 4.0f;
    
    Tensor *output = layer_forward(layer, input);
    
    assert(output != NULL);
    
    // Check sum is 1
    float sum = 0.0f;
    for (size_t i = 0; i < output->size; i++) {
        sum += output->data[i];
        assert(output->data[i] > 0.0f);
    }
    ASSERT_FLOAT_EQ(sum, 1.0f);
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

// ============ Sequential Layer Tests ============

TEST(layers_sequential_forward) {
    // Create a simple network: Linear -> ReLU -> Linear
    LayerConfig linear1_config = LINEAR(4, 3);
    LayerConfig relu_config = RELU();
    LayerConfig linear2_config = LINEAR(3, 2);
    
    Layer *linear1 = layer_create(linear1_config);
    Layer *relu = layer_create(relu_config);
    Layer *linear2 = layer_create(linear2_config);
    
    tensor_fill(linear1->weights, 1.0f);
    tensor_fill(linear1->bias, 0.0f);
    tensor_fill(linear2->weights, 1.0f);
    tensor_fill(linear2->bias, 0.0f);
    
    size_t input_shape[] = {4};
    Tensor *input = tensor_ones(input_shape, 1);
    
    // Forward through layers
    Tensor *out1 = layer_forward(linear1, input);
    Tensor *out2 = layer_forward(relu, out1);
    Tensor *out3 = layer_forward(linear2, out2);
    
    assert(out3 != NULL);
    assert(out3->shape[0] == 2);
    
    tensor_free(input);
    tensor_free(out1);
    tensor_free(out2);
    tensor_free(out3);
    layer_free(linear1);
    layer_free(relu);
    layer_free(linear2);
}

// ============ Parameter Management Tests ============

TEST(layer_get_parameters_linear) {
    LayerConfig config = LINEAR(5, 3);
    Layer *layer = layer_create(config);
    
    size_t num_params;
    Tensor **params = layer_get_parameters(layer, &num_params);
    
    assert(params != NULL);
    assert(num_params == 2);
    assert(params[0] == layer->weights);
    assert(params[1] == layer->bias);
    
    layer_free(layer);
}

TEST(layer_get_parameters_activation) {
    LayerConfig config = RELU();
    Layer *layer = layer_create(config);
    
    size_t num_params;
    Tensor **params = layer_get_parameters(layer, &num_params);
    
    // Activation layers have no parameters
    assert(num_params == 0);
    
    layer_free(layer);
}

TEST(layer_zero_grad) {
    LayerConfig config = LINEAR(3, 2);
    Layer *layer = layer_create(config);
    
    // Set requires_grad and allocate gradients
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    
    layer->weights->grad = (float*)calloc(layer->weights->size, sizeof(float));
    layer->bias->grad = (float*)calloc(layer->bias->size, sizeof(float));
    
    // Fill gradients with some values
    for (size_t i = 0; i < layer->weights->size; i++) {
        layer->weights->grad[i] = 1.0f;
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        layer->bias->grad[i] = 2.0f;
    }
    
    // Zero gradients
    layer_zero_grad(layer);
    
    // Check all gradients are zero
    for (size_t i = 0; i < layer->weights->size; i++) {
        ASSERT_FLOAT_EQ(layer->weights->grad[i], 0.0f);
    }
    for (size_t i = 0; i < layer->bias->size; i++) {
        ASSERT_FLOAT_EQ(layer->bias->grad[i], 0.0f);
    }
    
    layer_free(layer);
}

// ============ Autograd Integration Tests ============

TEST(layer_linear_autograd) {
    LayerConfig config = LINEAR(3, 2);
    Layer *layer = layer_create(config);
    
    tensor_fill(layer->weights, 1.0f);
    tensor_fill(layer->bias, 0.5f);
    
    tensor_set_requires_grad(layer->weights, 1);
    tensor_set_requires_grad(layer->bias, 1);
    
    size_t input_shape[] = {3};
    Tensor *input = tensor_ones(input_shape, 1);
    tensor_set_requires_grad(input, 1);
    
    Tensor *output = layer_forward(layer, input);
    
    // Output should have requires_grad set
    assert(output->requires_grad == 1);
    assert(output->inputs != NULL);
    
    tensor_free(input);
    tensor_free(output);
    layer_free(layer);
}

// ============ Edge Cases and Error Handling ============

TEST(layer_forward_null_input) {
    LayerConfig config = LINEAR(3, 2);
    Layer *layer = layer_create(config);
    
    Tensor *output = layer_forward(layer, NULL);
    assert(output == NULL);
    
    layer_free(layer);
}

TEST(layer_forward_null_layer) {
    size_t shape[] = {3};
    Tensor *input = tensor_ones(shape, 1);
    
    Tensor *output = layer_forward(NULL, input);
    assert(output == NULL);
    
    tensor_free(input);
}

TEST(layer_free_null) {
    layer_free(NULL);  // Should not crash
}

TEST(layer_get_parameters_null) {
    size_t num_params;
    Tensor **params = layer_get_parameters(NULL, &num_params);
    assert(params == NULL);
    
    LayerConfig config = LINEAR(2, 2);
    Layer *layer = layer_create(config);
    params = layer_get_parameters(layer, NULL);
    assert(params == NULL);
    
    layer_free(layer);
}

TEST(layer_zero_grad_null) {
    layer_zero_grad(NULL);  // Should not crash
}

// ============ Layer Dimension Tests ============

TEST(layer_linear_various_dimensions) {
    size_t configs[][2] = {
        {1, 1},
        {10, 1},
        {1, 10},
        {100, 50},
        {784, 128}
    };
    
    for (size_t i = 0; i < 5; i++) {
        LayerConfig config = LINEAR(configs[i][0], configs[i][1]);
        Layer *layer = layer_create(config);
        
        assert(layer != NULL);
        assert(layer->weights->shape[0] == configs[i][0]);
        assert(layer->weights->shape[1] == configs[i][1]);
        assert(layer->bias->shape[0] == configs[i][1]);
        
        layer_free(layer);
    }
}

TEST(layer_activation_preserves_shape) {
    LayerConfig configs[] = {RELU(), SIGMOID(), TANH(), SOFTMAX()};
    
    for (size_t i = 0; i < 4; i++) {
        Layer *layer = layer_create(configs[i]);
        
        size_t shape[] = {10};
        Tensor *input = tensor_randn(shape, 1, 42);
        
        Tensor *output = layer_forward(layer, input);
        
        assert(output != NULL);
        assert(output->ndim == input->ndim);
        assert(output->shape[0] == input->shape[0]);
        assert(output->size == input->size);
        
        tensor_free(input);
        tensor_free(output);
        layer_free(layer);
    }
}

TEST(layer_linear_weight_initialization) {
    LayerConfig config = LINEAR(10, 10);
    Layer *layer = layer_create(config);
    
    // Weights should be initialized (not all zeros)
    int has_nonzero = 0;
    for (size_t i = 0; i < layer->weights->size; i++) {
        if (fabsf(layer->weights->data[i]) > EPSILON) {
            has_nonzero = 1;
            break;
        }
    }
    assert(has_nonzero);
    
    // Bias should be initialized to zeros
    for (size_t i = 0; i < layer->bias->size; i++) {
        ASSERT_FLOAT_EQ(layer->bias->data[i], 0.0f);
    }
    
    layer_free(layer);
}

int main() {
    printf("\n=== Running Layer Unit Tests ===\n\n");
    
    // Layer creation
    RUN_TEST(layer_create_linear);
    RUN_TEST(layer_create_relu);
    RUN_TEST(layer_create_sigmoid);
    RUN_TEST(layer_create_tanh);
    RUN_TEST(layer_create_softmax);
    
    // Forward pass
    RUN_TEST(layer_forward_linear_single);
    RUN_TEST(layer_forward_linear_batch);
    RUN_TEST(layer_forward_relu);
    RUN_TEST(layer_forward_sigmoid);
    RUN_TEST(layer_forward_tanh);
    RUN_TEST(layer_forward_softmax);
    
    // Sequential layers
    RUN_TEST(layers_sequential_forward);
    
    // Parameter management
    RUN_TEST(layer_get_parameters_linear);
    RUN_TEST(layer_get_parameters_activation);
    RUN_TEST(layer_zero_grad);
    
    // Autograd integration
    RUN_TEST(layer_linear_autograd);
    
    // Edge cases
    RUN_TEST(layer_forward_null_input);
    RUN_TEST(layer_forward_null_layer);
    RUN_TEST(layer_free_null);
    RUN_TEST(layer_get_parameters_null);
    RUN_TEST(layer_zero_grad_null);
    
    // Dimension tests
    RUN_TEST(layer_linear_various_dimensions);
    RUN_TEST(layer_activation_preserves_shape);
    RUN_TEST(layer_linear_weight_initialization);
    
    printf("\n=== All Layer Tests Passed! ===\n\n");
    return 0;
}
