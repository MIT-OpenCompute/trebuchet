/*
 * AI generated Unit tests for autograd.c
 * Tests gradient computation for all operations and activation functions
 */

#include "lessbasednn/ops.h"
#include "lessbasednn/autograd.h"
#include "lessbasednn/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define EPSILON 1e-4
#define GRAD_EPSILON 1e-3
#define ASSERT_FLOAT_EQ(a, b) assert(fabsf((a) - (b)) < EPSILON)
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    printf("Running test_%s...\n", #name); \
    test_##name(); \
    printf("  ✓ test_%s passed\n", #name); \
} while(0)

// Numerical gradient checking helper
static float numerical_gradient(Tensor *T, size_t idx, Tensor *(*forward_fn)(void*), void *ctx) {
    float original = T->data[idx];
    float h = 1e-5f;
    
    T->data[idx] = original + h;
    Tensor *out_plus = forward_fn(ctx);
    float f_plus = out_plus->data[0];
    tensor_free(out_plus);
    
    T->data[idx] = original - h;
    Tensor *out_minus = forward_fn(ctx);
    float f_minus = out_minus->data[0];
    tensor_free(out_minus);
    
    T->data[idx] = original;
    return (f_plus - f_minus) / (2.0f * h);
}

// ============ Backward Add Tests ============

TEST(backward_add_basic) {
    size_t shape[] = {2, 2};
    Tensor *A = tensor_create(shape, 2);
    Tensor *B = tensor_create(shape, 2);
    
    for (size_t i = 0; i < 4; i++) {
        A->data[i] = (float)i;
        B->data[i] = (float)(i + 1);
    }
    
    tensor_set_requires_grad(A, 1);
    tensor_set_requires_grad(B, 1);
    
    Tensor *C = tensor_add(A, B);
    
    // Manually set gradient for C
    C->grad = (float*)calloc(C->size, sizeof(float));
    for (size_t i = 0; i < C->size; i++) {
        C->grad[i] = 1.0f;
    }
    
    backward_add(C);
    
    // Gradient of add: dC/dA = 1, dC/dB = 1
    assert(A->grad != NULL);
    assert(B->grad != NULL);
    for (size_t i = 0; i < 4; i++) {
        ASSERT_FLOAT_EQ(A->grad[i], 1.0f);
        ASSERT_FLOAT_EQ(B->grad[i], 1.0f);
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(backward_add_broadcasting) {
    size_t shape_a[] = {3, 4};
    size_t shape_b[] = {4};
    
    Tensor *A = tensor_ones(shape_a, 2);
    Tensor *B = tensor_ones(shape_b, 1);
    
    tensor_set_requires_grad(A, 1);
    tensor_set_requires_grad(B, 1);
    
    Tensor *C = tensor_add(A, B);
    
    C->grad = (float*)calloc(C->size, sizeof(float));
    for (size_t i = 0; i < C->size; i++) {
        C->grad[i] = 1.0f;
    }
    
    backward_add(C);
    
    // A gradient should be all 1s
    for (size_t i = 0; i < A->size; i++) {
        ASSERT_FLOAT_EQ(A->grad[i], 1.0f);
    }
    
    // B gradient should accumulate across rows
    for (size_t i = 0; i < B->size; i++) {
        ASSERT_FLOAT_EQ(B->grad[i], 3.0f); // 3 rows
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

// ============ Backward Sub Tests ============

TEST(backward_sub_basic) {
    size_t shape[] = {3};
    Tensor *A = tensor_create(shape, 1);
    Tensor *B = tensor_create(shape, 1);
    
    A->data[0] = 5.0f; A->data[1] = 3.0f; A->data[2] = 1.0f;
    B->data[0] = 2.0f; B->data[1] = 1.0f; B->data[2] = 0.5f;
    
    tensor_set_requires_grad(A, 1);
    tensor_set_requires_grad(B, 1);
    
    Tensor *C = tensor_sub(A, B);
    
    C->grad = (float*)calloc(C->size, sizeof(float));
    for (size_t i = 0; i < C->size; i++) {
        C->grad[i] = 1.0f;
    }
    
    backward_sub(C);
    
    // Gradient of sub: dC/dA = 1, dC/dB = -1
    for (size_t i = 0; i < 3; i++) {
        ASSERT_FLOAT_EQ(A->grad[i], 1.0f);
        ASSERT_FLOAT_EQ(B->grad[i], -1.0f);
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

// ============ Backward Mul Tests ============

TEST(backward_mul_basic) {
    size_t shape[] = {2};
    Tensor *A = tensor_create(shape, 1);
    Tensor *B = tensor_create(shape, 1);
    
    A->data[0] = 3.0f; A->data[1] = 4.0f;
    B->data[0] = 2.0f; B->data[1] = 5.0f;
    
    tensor_set_requires_grad(A, 1);
    tensor_set_requires_grad(B, 1);
    
    Tensor *C = tensor_mul(A, B);
    
    C->grad = (float*)calloc(C->size, sizeof(float));
    C->grad[0] = 1.0f; C->grad[1] = 1.0f;
    
    backward_mul(C);
    
    // Gradient of mul: dC/dA = B, dC/dB = A
    ASSERT_FLOAT_EQ(A->grad[0], 2.0f);
    ASSERT_FLOAT_EQ(A->grad[1], 5.0f);
    ASSERT_FLOAT_EQ(B->grad[0], 3.0f);
    ASSERT_FLOAT_EQ(B->grad[1], 4.0f);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

// ============ Backward Matmul Tests ============

TEST(backward_matmul_vector_vector) {
    size_t shape[] = {3};
    Tensor *A = tensor_create(shape, 1);
    Tensor *B = tensor_create(shape, 1);
    
    A->data[0] = 1.0f; A->data[1] = 2.0f; A->data[2] = 3.0f;
    B->data[0] = 4.0f; B->data[1] = 5.0f; B->data[2] = 6.0f;
    
    tensor_set_requires_grad(A, 1);
    tensor_set_requires_grad(B, 1);
    
    Tensor *C = tensor_matmul(A, B);
    
    C->grad = (float*)calloc(C->size, sizeof(float));
    C->grad[0] = 1.0f;
    
    backward_matmul(C);
    
    // For dot product: dC/dA[i] = B[i], dC/dB[i] = A[i]
    ASSERT_FLOAT_EQ(A->grad[0], 4.0f);
    ASSERT_FLOAT_EQ(A->grad[1], 5.0f);
    ASSERT_FLOAT_EQ(A->grad[2], 6.0f);
    
    ASSERT_FLOAT_EQ(B->grad[0], 1.0f);
    ASSERT_FLOAT_EQ(B->grad[1], 2.0f);
    ASSERT_FLOAT_EQ(B->grad[2], 3.0f);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(backward_matmul_matrix_vector) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3};
    
    Tensor *A = tensor_create(shape_a, 2);
    Tensor *B = tensor_create(shape_b, 1);
    
    for (size_t i = 0; i < A->size; i++) A->data[i] = (float)(i + 1);
    for (size_t i = 0; i < B->size; i++) B->data[i] = 1.0f;
    
    tensor_set_requires_grad(A, 1);
    tensor_set_requires_grad(B, 1);
    
    Tensor *C = tensor_matmul(A, B);
    
    C->grad = (float*)calloc(C->size, sizeof(float));
    C->grad[0] = 1.0f; C->grad[1] = 1.0f;
    
    backward_matmul(C);
    
    assert(A->grad != NULL);
    assert(B->grad != NULL);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(backward_matmul_matrix_matrix) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 2};
    
    Tensor *A = tensor_create(shape_a, 2);
    Tensor *B = tensor_create(shape_b, 2);
    
    for (size_t i = 0; i < A->size; i++) A->data[i] = 1.0f;
    for (size_t i = 0; i < B->size; i++) B->data[i] = 1.0f;
    
    tensor_set_requires_grad(A, 1);
    tensor_set_requires_grad(B, 1);
    
    Tensor *C = tensor_matmul(A, B);
    
    C->grad = (float*)calloc(C->size, sizeof(float));
    for (size_t i = 0; i < C->size; i++) C->grad[i] = 1.0f;
    
    backward_matmul(C);
    
    assert(A->grad != NULL);
    assert(B->grad != NULL);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

// ============ Backward Transpose Tests ============

TEST(backward_transpose_basic) {
    size_t shape[] = {2, 3};
    Tensor *A = tensor_create(shape, 2);
    
    for (size_t i = 0; i < A->size; i++) {
        A->data[i] = (float)i;
    }
    
    tensor_set_requires_grad(A, 1);
    
    Tensor *T = tensor_transpose(A);
    
    T->grad = (float*)calloc(T->size, sizeof(float));
    for (size_t i = 0; i < T->size; i++) {
        T->grad[i] = 1.0f;
    }
    
    backward_transpose(T);
    
    assert(A->grad != NULL);
    // All gradients should be 1 (transpose just reorders)
    for (size_t i = 0; i < A->size; i++) {
        ASSERT_FLOAT_EQ(A->grad[i], 1.0f);
    }
    
    tensor_free(A);
    tensor_free(T);
}

// ============ Backward ReLU Tests ============

TEST(backward_relu_basic) {
    size_t shape[] = {5};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = -2.0f;
    Z->data[1] = -0.5f;
    Z->data[2] = 0.0f;
    Z->data[3] = 1.0f;
    Z->data[4] = 3.0f;
    
    tensor_set_requires_grad(Z, 1);
    
    Tensor *A = tensor_relu(Z);
    
    A->grad = (float*)calloc(A->size, sizeof(float));
    for (size_t i = 0; i < A->size; i++) {
        A->grad[i] = 1.0f;
    }
    
    backward_relu(A);
    
    // Gradient is 1 where input > 0, 0 otherwise
    ASSERT_FLOAT_EQ(Z->grad[0], 0.0f);
    ASSERT_FLOAT_EQ(Z->grad[1], 0.0f);
    ASSERT_FLOAT_EQ(Z->grad[2], 0.0f);
    ASSERT_FLOAT_EQ(Z->grad[3], 1.0f);
    ASSERT_FLOAT_EQ(Z->grad[4], 1.0f);
    
    tensor_free(Z);
    tensor_free(A);
}

// ============ Backward Sigmoid Tests ============

TEST(backward_sigmoid_basic) {
    size_t shape[] = {3};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = -1.0f;
    Z->data[1] = 0.0f;
    Z->data[2] = 1.0f;
    
    tensor_set_requires_grad(Z, 1);
    
    Tensor *A = tensor_sigmoid(Z);
    
    A->grad = (float*)calloc(A->size, sizeof(float));
    for (size_t i = 0; i < A->size; i++) {
        A->grad[i] = 1.0f;
    }
    
    backward_sigmoid(A);
    
    // Gradient: sigmoid * (1 - sigmoid)
    for (size_t i = 0; i < 3; i++) {
        float sig = A->data[i];
        float expected_grad = sig * (1.0f - sig);
        assert(fabsf(Z->grad[i] - expected_grad) < GRAD_EPSILON);
    }
    
    tensor_free(Z);
    tensor_free(A);
}

TEST(backward_sigmoid_gradient_vanishing) {
    size_t shape[] = {2};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = -10.0f; // Very negative
    Z->data[1] = 10.0f;  // Very positive
    
    tensor_set_requires_grad(Z, 1);
    
    Tensor *A = tensor_sigmoid(Z);
    
    A->grad = (float*)calloc(A->size, sizeof(float));
    A->grad[0] = 1.0f; A->grad[1] = 1.0f;
    
    backward_sigmoid(A);
    
    // Gradients should be very small (vanishing gradient)
    assert(fabsf(Z->grad[0]) < 0.01f);
    assert(fabsf(Z->grad[1]) < 0.01f);
    
    tensor_free(Z);
    tensor_free(A);
}

// ============ Backward Tanh Tests ============

TEST(backward_tanh_basic) {
    size_t shape[] = {3};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = -1.0f;
    Z->data[1] = 0.0f;
    Z->data[2] = 1.0f;
    
    tensor_set_requires_grad(Z, 1);
    
    Tensor *A = tensor_tanh(Z);
    
    A->grad = (float*)calloc(A->size, sizeof(float));
    for (size_t i = 0; i < A->size; i++) {
        A->grad[i] = 1.0f;
    }
    
    backward_tanh(A);
    
    // Gradient: 1 - tanh^2
    for (size_t i = 0; i < 3; i++) {
        float t = A->data[i];
        float expected_grad = 1.0f - t * t;
        assert(fabsf(Z->grad[i] - expected_grad) < GRAD_EPSILON);
    }
    
    tensor_free(Z);
    tensor_free(A);
}

// ============ Backward Softmax Tests ============

TEST(backward_softmax_basic) {
    size_t shape[] = {3};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = 1.0f;
    Z->data[1] = 2.0f;
    Z->data[2] = 3.0f;
    
    tensor_set_requires_grad(Z, 1);
    
    Tensor *A = tensor_softmax(Z);
    
    A->grad = (float*)calloc(A->size, sizeof(float));
    A->grad[0] = 1.0f; A->grad[1] = 0.0f; A->grad[2] = 0.0f;
    
    backward_softmax(A);
    
    assert(Z->grad != NULL);
    
    tensor_free(Z);
    tensor_free(A);
}

// ============ End-to-End Gradient Tests ============

TEST(gradient_flow_simple_network) {
    // Simple computation: C = ReLU(A * B)
    size_t shape[] = {3};
    Tensor *A = tensor_create(shape, 1);
    Tensor *B = tensor_create(shape, 1);
    
    A->data[0] = 1.0f; A->data[1] = -2.0f; A->data[2] = 3.0f;
    B->data[0] = 2.0f; B->data[1] = 1.0f; B->data[2] = 0.5f;
    
    tensor_set_requires_grad(A, 1);
    tensor_set_requires_grad(B, 1);
    
    Tensor *C = tensor_mul(A, B);
    Tensor *D = tensor_relu(C);
    
    D->grad = (float*)calloc(D->size, sizeof(float));
    for (size_t i = 0; i < D->size; i++) {
        D->grad[i] = 1.0f;
    }
    
    backward_relu(D);
    backward_mul(C);
    
    assert(A->grad != NULL);
    assert(B->grad != NULL);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    tensor_free(D);
}

TEST(gradient_accumulation) {
    size_t shape[] = {2};
    Tensor *A = tensor_create(shape, 1);
    
    A->data[0] = 1.0f; A->data[1] = 2.0f;
    
    tensor_set_requires_grad(A, 1);
    
    // Create two operations using A
    Tensor *B = tensor_create(shape, 1);
    B->data[0] = 2.0f; B->data[1] = 3.0f;
    tensor_set_requires_grad(B, 1);
    
    Tensor *C1 = tensor_add(A, B);
    Tensor *C2 = tensor_mul(A, B);
    
    // Set gradients
    C1->grad = (float*)calloc(C1->size, sizeof(float));
    C2->grad = (float*)calloc(C2->size, sizeof(float));
    for (size_t i = 0; i < 2; i++) {
        C1->grad[i] = 1.0f;
        C2->grad[i] = 1.0f;
    }
    
    // Backward passes should accumulate gradients
    backward_add(C1);
    backward_mul(C2);
    
    // A gradient should be sum from both operations
    // From add: 1.0, From mul: B
    ASSERT_FLOAT_EQ(A->grad[0], 1.0f + 2.0f);
    ASSERT_FLOAT_EQ(A->grad[1], 1.0f + 3.0f);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C1);
    tensor_free(C2);
}

TEST(no_gradient_when_not_required) {
    size_t shape[] = {2, 2};
    Tensor *A = tensor_ones(shape, 2);
    Tensor *B = tensor_ones(shape, 2);
    
    // Don't set requires_grad
    
    Tensor *C = tensor_add(A, B);
    
    // Should not have autograd setup
    assert(C->requires_grad == 0);
    assert(C->inputs == NULL);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(backward_with_null_inputs) {
    size_t shape[] = {2};
    Tensor *T = tensor_create(shape, 1);
    T->data[0] = 1.0f; T->data[1] = 2.0f;
    
    T->grad = (float*)calloc(T->size, sizeof(float));
    T->grad[0] = 1.0f; T->grad[1] = 1.0f;
    
    T->inputs = NULL;
    T->num_inputs = 0;
    
    // These should not crash
    backward_add(T);
    backward_sub(T);
    backward_mul(T);
    
    tensor_free(T);
}

int main() {
    printf("\n=== Running Autograd Unit Tests ===\n\n");
    
    // Backward operations
    RUN_TEST(backward_add_basic);
    RUN_TEST(backward_add_broadcasting);
    RUN_TEST(backward_sub_basic);
    RUN_TEST(backward_mul_basic);
    
    // Backward matmul
    RUN_TEST(backward_matmul_vector_vector);
    RUN_TEST(backward_matmul_matrix_vector);
    RUN_TEST(backward_matmul_matrix_matrix);
    
    // Backward transpose
    RUN_TEST(backward_transpose_basic);
    
    // Backward activations
    RUN_TEST(backward_relu_basic);
    RUN_TEST(backward_sigmoid_basic);
    RUN_TEST(backward_sigmoid_gradient_vanishing);
    RUN_TEST(backward_tanh_basic);
    RUN_TEST(backward_softmax_basic);
    
    // End-to-end tests
    RUN_TEST(gradient_flow_simple_network);
    RUN_TEST(gradient_accumulation);
    RUN_TEST(no_gradient_when_not_required);
    RUN_TEST(backward_with_null_inputs);
    
    printf("\n=== All Autograd Tests Passed! ===\n\n");
    return 0;
}
