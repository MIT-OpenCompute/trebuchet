/*
 * AI generated Unit tests for ops.c
 * Tests tensor operations, activations, loss functions, and utilities
 */

#include "lessbasednn/ops.h"
#include "lessbasednn/tensor.h"
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

// ============ Elementwise Operations Tests ============

TEST(tensor_add_basic) {
    size_t shape[] = {2, 3};
    Tensor *A = tensor_create(shape, 2);
    Tensor *B = tensor_create(shape, 2);
    
    for (size_t i = 0; i < A->size; i++) {
        A->data[i] = (float)i;
        B->data[i] = (float)(i * 2);
    }
    
    Tensor *C = tensor_add(A, B);
    
    assert(C != NULL);
    for (size_t i = 0; i < C->size; i++) {
        ASSERT_FLOAT_EQ(C->data[i], A->data[i] + B->data[i]);
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(tensor_add_broadcasting) {
    // Test 2D + 1D broadcasting (matrix + vector)
    size_t shape_a[] = {3, 4};
    size_t shape_b[] = {4};
    
    Tensor *A = tensor_create(shape_a, 2);
    Tensor *B = tensor_create(shape_b, 1);
    
    for (size_t i = 0; i < A->size; i++) A->data[i] = 1.0f;
    for (size_t i = 0; i < B->size; i++) B->data[i] = (float)i;
    
    Tensor *C = tensor_add(A, B);
    
    assert(C != NULL);
    // Check broadcasting worked correctly
    for (size_t i = 0; i < A->shape[0]; i++) {
        for (size_t j = 0; j < A->shape[1]; j++) {
            ASSERT_FLOAT_EQ(C->data[i * A->shape[1] + j], 1.0f + (float)j);
        }
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(tensor_add_autograd) {
    size_t shape[] = {2, 2};
    Tensor *A = tensor_ones(shape, 2);
    Tensor *B = tensor_ones(shape, 2);
    
    tensor_set_requires_grad(A, 1);
    tensor_set_requires_grad(B, 1);
    
    Tensor *C = tensor_add(A, B);
    
    assert(C->requires_grad == 1);
    assert(C->op == OP_ADD);
    assert(C->num_inputs == 2);
    assert(C->inputs[0] == A);
    assert(C->inputs[1] == B);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(tensor_sub_basic) {
    size_t shape[] = {3, 3};
    Tensor *A = tensor_create(shape, 2);
    Tensor *B = tensor_create(shape, 2);
    
    for (size_t i = 0; i < A->size; i++) {
        A->data[i] = (float)(i + 10);
        B->data[i] = (float)i;
    }
    
    Tensor *C = tensor_sub(A, B);
    
    assert(C != NULL);
    for (size_t i = 0; i < C->size; i++) {
        ASSERT_FLOAT_EQ(C->data[i], 10.0f);
    }
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(tensor_mul_basic) {
    size_t shape[] = {2, 2};
    Tensor *A = tensor_create(shape, 2);
    Tensor *B = tensor_create(shape, 2);
    
    A->data[0] = 2.0f; A->data[1] = 3.0f;
    A->data[2] = 4.0f; A->data[3] = 5.0f;
    
    B->data[0] = 1.0f; B->data[1] = 2.0f;
    B->data[2] = 3.0f; B->data[3] = 4.0f;
    
    Tensor *C = tensor_mul(A, B);
    
    assert(C != NULL);
    ASSERT_FLOAT_EQ(C->data[0], 2.0f);
    ASSERT_FLOAT_EQ(C->data[1], 6.0f);
    ASSERT_FLOAT_EQ(C->data[2], 12.0f);
    ASSERT_FLOAT_EQ(C->data[3], 20.0f);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

// ============ Matrix Multiplication Tests ============

TEST(tensor_matmul_vector_vector) {
    size_t shape[] = {3};
    Tensor *A = tensor_create(shape, 1);
    Tensor *B = tensor_create(shape, 1);
    
    A->data[0] = 1.0f; A->data[1] = 2.0f; A->data[2] = 3.0f;
    B->data[0] = 4.0f; B->data[1] = 5.0f; B->data[2] = 6.0f;
    
    Tensor *C = tensor_matmul(A, B);
    
    assert(C != NULL);
    assert(C->ndim == 1);
    assert(C->size == 1);
    ASSERT_FLOAT_EQ(C->data[0], 32.0f); // 1*4 + 2*5 + 3*6 = 32
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(tensor_matmul_matrix_vector) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3};
    
    Tensor *A = tensor_create(shape_a, 2);
    Tensor *B = tensor_create(shape_b, 1);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    A->data[0] = 1.0f; A->data[1] = 2.0f; A->data[2] = 3.0f;
    A->data[3] = 4.0f; A->data[4] = 5.0f; A->data[5] = 6.0f;
    
    // B = [1, 2, 3]
    B->data[0] = 1.0f; B->data[1] = 2.0f; B->data[2] = 3.0f;
    
    Tensor *C = tensor_matmul(A, B);
    
    assert(C != NULL);
    assert(C->ndim == 1);
    assert(C->shape[0] == 2);
    ASSERT_FLOAT_EQ(C->data[0], 14.0f); // 1*1 + 2*2 + 3*3 = 14
    ASSERT_FLOAT_EQ(C->data[1], 32.0f); // 4*1 + 5*2 + 6*3 = 32
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(tensor_matmul_vector_matrix) {
    size_t shape_a[] = {3};
    size_t shape_b[] = {3, 2};
    
    Tensor *A = tensor_create(shape_a, 1);
    Tensor *B = tensor_create(shape_b, 2);
    
    // A = [1, 2, 3]
    A->data[0] = 1.0f; A->data[1] = 2.0f; A->data[2] = 3.0f;
    
    // B = [[1, 2], [3, 4], [5, 6]]
    B->data[0] = 1.0f; B->data[1] = 2.0f;
    B->data[2] = 3.0f; B->data[3] = 4.0f;
    B->data[4] = 5.0f; B->data[5] = 6.0f;
    
    Tensor *C = tensor_matmul(A, B);
    
    assert(C != NULL);
    assert(C->ndim == 1);
    assert(C->shape[0] == 2);
    ASSERT_FLOAT_EQ(C->data[0], 22.0f); // 1*1 + 2*3 + 3*5 = 22
    ASSERT_FLOAT_EQ(C->data[1], 28.0f); // 1*2 + 2*4 + 3*6 = 28
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(tensor_matmul_matrix_matrix) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 2};
    
    Tensor *A = tensor_create(shape_a, 2);
    Tensor *B = tensor_create(shape_b, 2);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    A->data[0] = 1.0f; A->data[1] = 2.0f; A->data[2] = 3.0f;
    A->data[3] = 4.0f; A->data[4] = 5.0f; A->data[5] = 6.0f;
    
    // B = [[7, 8], [9, 10], [11, 12]]
    B->data[0] = 7.0f; B->data[1] = 8.0f;
    B->data[2] = 9.0f; B->data[3] = 10.0f;
    B->data[4] = 11.0f; B->data[5] = 12.0f;
    
    Tensor *C = tensor_matmul(A, B);
    
    assert(C != NULL);
    assert(C->ndim == 2);
    assert(C->shape[0] == 2);
    assert(C->shape[1] == 2);
    
    // C = [[58, 64], [139, 154]]
    ASSERT_FLOAT_EQ(C->data[0], 58.0f);
    ASSERT_FLOAT_EQ(C->data[1], 64.0f);
    ASSERT_FLOAT_EQ(C->data[2], 139.0f);
    ASSERT_FLOAT_EQ(C->data[3], 154.0f);
    
    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
}

TEST(tensor_matmul_incompatible_shapes) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {4, 2};
    
    Tensor *A = tensor_ones(shape_a, 2);
    Tensor *B = tensor_ones(shape_b, 2);
    
    Tensor *C = tensor_matmul(A, B);
    
    assert(C == NULL); // Should fail due to incompatible shapes
    
    tensor_free(A);
    tensor_free(B);
}

TEST(tensor_matmul_identity) {
    size_t shape[] = {3, 3};
    Tensor *A = tensor_create(shape, 2);
    Tensor *I = tensor_zeroes(shape, 2);
    
    // Set A to some values
    for (size_t i = 0; i < 9; i++) A->data[i] = (float)(i + 1);
    
    // Set I to identity matrix
    I->data[0] = 1.0f; I->data[4] = 1.0f; I->data[8] = 1.0f;
    
    Tensor *C = tensor_matmul(A, I);
    
    // A * I should equal A
    assert(C != NULL);
    for (size_t i = 0; i < A->size; i++) {
        ASSERT_FLOAT_EQ(C->data[i], A->data[i]);
    }
    
    tensor_free(A);
    tensor_free(I);
    tensor_free(C);
}

// ============ Transpose Tests ============

TEST(tensor_transpose_basic) {
    size_t shape[] = {2, 3};
    Tensor *A = tensor_create(shape, 2);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    for (size_t i = 0; i < A->size; i++) {
        A->data[i] = (float)(i + 1);
    }
    
    Tensor *T = tensor_transpose(A);
    
    assert(T != NULL);
    assert(T->ndim == 2);
    assert(T->shape[0] == 3);
    assert(T->shape[1] == 2);
    
    // T should be [[1, 4], [2, 5], [3, 6]]
    ASSERT_FLOAT_EQ(T->data[0], 1.0f); ASSERT_FLOAT_EQ(T->data[1], 4.0f);
    ASSERT_FLOAT_EQ(T->data[2], 2.0f); ASSERT_FLOAT_EQ(T->data[3], 5.0f);
    ASSERT_FLOAT_EQ(T->data[4], 3.0f); ASSERT_FLOAT_EQ(T->data[5], 6.0f);
    
    tensor_free(A);
    tensor_free(T);
}

TEST(tensor_transpose_square) {
    size_t shape[] = {3, 3};
    Tensor *A = tensor_create(shape, 2);
    
    for (size_t i = 0; i < A->size; i++) {
        A->data[i] = (float)i;
    }
    
    Tensor *T = tensor_transpose(A);
    
    assert(T != NULL);
    assert(T->shape[0] == 3);
    assert(T->shape[1] == 3);
    
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            ASSERT_FLOAT_EQ(T->data[j * 3 + i], A->data[i * 3 + j]);
        }
    }
    
    tensor_free(A);
    tensor_free(T);
}

TEST(tensor_transpose_double) {
    size_t shape[] = {2, 3};
    Tensor *A = tensor_randn(shape, 2, 42);
    
    Tensor *T1 = tensor_transpose(A);
    Tensor *T2 = tensor_transpose(T1);
    
    // Double transpose should equal original
    assert(T2->shape[0] == A->shape[0]);
    assert(T2->shape[1] == A->shape[1]);
    
    for (size_t i = 0; i < A->size; i++) {
        ASSERT_FLOAT_EQ(T2->data[i], A->data[i]);
    }
    
    tensor_free(A);
    tensor_free(T1);
    tensor_free(T2);
}

// ============ Activation Function Tests ============

TEST(tensor_relu_basic) {
    size_t shape[] = {4};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = -2.0f;
    Z->data[1] = -0.5f;
    Z->data[2] = 0.0f;
    Z->data[3] = 3.0f;
    
    Tensor *A = tensor_relu(Z);
    
    assert(A != NULL);
    ASSERT_FLOAT_EQ(A->data[0], 0.0f);
    ASSERT_FLOAT_EQ(A->data[1], 0.0f);
    ASSERT_FLOAT_EQ(A->data[2], 0.0f);
    ASSERT_FLOAT_EQ(A->data[3], 3.0f);
    
    tensor_free(Z);
    tensor_free(A);
}

TEST(tensor_relu_all_positive) {
    size_t shape[] = {3, 3};
    Tensor *Z = tensor_ones(shape, 2);
    
    for (size_t i = 0; i < Z->size; i++) {
        Z->data[i] = (float)(i + 1);
    }
    
    Tensor *A = tensor_relu(Z);
    
    // All positive values should remain unchanged
    for (size_t i = 0; i < A->size; i++) {
        ASSERT_FLOAT_EQ(A->data[i], Z->data[i]);
    }
    
    tensor_free(Z);
    tensor_free(A);
}

TEST(tensor_sigmoid_basic) {
    size_t shape[] = {3};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = 0.0f;
    Z->data[1] = 1000.0f; // Large positive
    Z->data[2] = -1000.0f; // Large negative
    
    Tensor *A = tensor_sigmoid(Z);
    
    assert(A != NULL);
    ASSERT_FLOAT_EQ(A->data[0], 0.5f);
    ASSERT_FLOAT_EQ(A->data[1], 1.0f); // Should be close to 1
    ASSERT_FLOAT_EQ(A->data[2], 0.0f); // Should be close to 0
    
    tensor_free(Z);
    tensor_free(A);
}

TEST(tensor_sigmoid_range) {
    size_t shape[] = {5};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = -2.0f;
    Z->data[1] = -1.0f;
    Z->data[2] = 0.0f;
    Z->data[3] = 1.0f;
    Z->data[4] = 2.0f;
    
    Tensor *A = tensor_sigmoid(Z);
    
    // All outputs should be in (0, 1)
    for (size_t i = 0; i < A->size; i++) {
        assert(A->data[i] > 0.0f && A->data[i] < 1.0f);
    }
    
    tensor_free(Z);
    tensor_free(A);
}

TEST(tensor_tanh_basic) {
    size_t shape[] = {3};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = 0.0f;
    Z->data[1] = 1.0f;
    Z->data[2] = -1.0f;
    
    Tensor *A = tensor_tanh(Z);
    
    assert(A != NULL);
    ASSERT_FLOAT_EQ(A->data[0], 0.0f);
    ASSERT_FLOAT_EQ(A->data[1], tanhf(1.0f));
    ASSERT_FLOAT_EQ(A->data[2], tanhf(-1.0f));
    
    tensor_free(Z);
    tensor_free(A);
}

TEST(tensor_tanh_range) {
    size_t shape[] = {100};
    Tensor *Z = tensor_randn(shape, 1, 42);
    
    Tensor *A = tensor_tanh(Z);
    
    // All outputs should be in (-1, 1)
    for (size_t i = 0; i < A->size; i++) {
        assert(A->data[i] > -1.0f && A->data[i] < 1.0f);
    }
    
    tensor_free(Z);
    tensor_free(A);
}

TEST(tensor_softmax_basic) {
    size_t shape[] = {3};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = 1.0f;
    Z->data[1] = 2.0f;
    Z->data[2] = 3.0f;
    
    Tensor *A = tensor_softmax(Z);
    
    assert(A != NULL);
    
    // Check sum equals 1
    float sum = 0.0f;
    for (size_t i = 0; i < A->size; i++) {
        sum += A->data[i];
        assert(A->data[i] > 0.0f); // All positive
    }
    ASSERT_FLOAT_EQ(sum, 1.0f);
    
    tensor_free(Z);
    tensor_free(A);
}

TEST(tensor_softmax_numerical_stability) {
    size_t shape[] = {3};
    Tensor *Z = tensor_create(shape, 1);
    
    Z->data[0] = 1000.0f;
    Z->data[1] = 1001.0f;
    Z->data[2] = 1002.0f;
    
    Tensor *A = tensor_softmax(Z);
    
    assert(A != NULL);
    
    // Should still sum to 1 despite large values
    float sum = 0.0f;
    for (size_t i = 0; i < A->size; i++) {
        sum += A->data[i];
    }
    ASSERT_FLOAT_EQ(sum, 1.0f);
    
    tensor_free(Z);
    tensor_free(A);
}

// ============ Loss Function Tests ============

TEST(tensor_mse_basic) {
    size_t shape[] = {4};
    Tensor *pred = tensor_create(shape, 1);
    Tensor *target = tensor_create(shape, 1);
    
    pred->data[0] = 1.0f; pred->data[1] = 2.0f;
    pred->data[2] = 3.0f; pred->data[3] = 4.0f;
    
    target->data[0] = 1.0f; target->data[1] = 2.0f;
    target->data[2] = 3.0f; target->data[3] = 4.0f;
    
    Tensor *loss = tensor_mse(pred, target);
    
    assert(loss != NULL);
    ASSERT_FLOAT_EQ(loss->data[0], 0.0f); // Perfect prediction
    
    tensor_free(pred);
    tensor_free(target);
    tensor_free(loss);
}

TEST(tensor_mse_nonzero) {
    size_t shape[] = {2};
    Tensor *pred = tensor_create(shape, 1);
    Tensor *target = tensor_create(shape, 1);
    
    pred->data[0] = 0.0f; pred->data[1] = 2.0f;
    target->data[0] = 1.0f; target->data[1] = 1.0f;
    
    Tensor *loss = tensor_mse(pred, target);
    
    // MSE = ((0-1)^2 + (2-1)^2) / 2 = (1 + 1) / 2 = 1.0
    ASSERT_FLOAT_EQ(loss->data[0], 1.0f);
    
    tensor_free(pred);
    tensor_free(target);
    tensor_free(loss);
}

TEST(tensor_cross_entropy_basic) {
    size_t shape[] = {3};
    Tensor *pred = tensor_create(shape, 1);
    Tensor *target = tensor_create(shape, 1);
    
    pred->data[0] = 0.7f; pred->data[1] = 0.2f; pred->data[2] = 0.1f;
    target->data[0] = 1.0f; target->data[1] = 0.0f; target->data[2] = 0.0f;
    
    Tensor *loss = tensor_cross_entropy(pred, target);
    
    assert(loss != NULL);
    // Loss should be -log(0.7) / 3
    float expected = -logf(0.7f) / 3.0f;
    ASSERT_FLOAT_EQ(loss->data[0], expected);
    
    tensor_free(pred);
    tensor_free(target);
    tensor_free(loss);
}

TEST(tensor_binary_cross_entropy_basic) {
    size_t shape[] = {4};
    Tensor *pred = tensor_create(shape, 1);
    Tensor *target = tensor_create(shape, 1);
    
    pred->data[0] = 0.9f; pred->data[1] = 0.8f;
    pred->data[2] = 0.1f; pred->data[3] = 0.2f;
    
    target->data[0] = 1.0f; target->data[1] = 1.0f;
    target->data[2] = 0.0f; target->data[3] = 0.0f;
    
    Tensor *loss = tensor_binary_cross_entropy(pred, target);
    
    assert(loss != NULL);
    assert(loss->data[0] > 0.0f); // Loss should be positive
    
    tensor_free(pred);
    tensor_free(target);
    tensor_free(loss);
}

// ============ Tensor Slice Tests ============

TEST(tensor_slice_basic) {
    size_t shape[] = {5, 3};
    Tensor *T = tensor_create(shape, 2);
    
    for (size_t i = 0; i < T->size; i++) {
        T->data[i] = (float)i;
    }
    
    Tensor *slice = tensor_slice(T, 1, 4);
    
    assert(slice != NULL);
    assert(slice->ndim == 2);
    assert(slice->shape[0] == 3);
    assert(slice->shape[1] == 3);
    assert(slice->size == 9);
    assert(slice->owns_data == 0); // Doesn't own data
    
    // Check that slice points to correct data
    ASSERT_FLOAT_EQ(slice->data[0], T->data[3]);
    
    tensor_free(slice); // Only free the tensor struct, not data
    tensor_free(T);
}

TEST(tensor_slice_1d) {
    size_t shape[] = {10};
    Tensor *T = tensor_create(shape, 1);
    
    for (size_t i = 0; i < T->size; i++) {
        T->data[i] = (float)i;
    }
    
    Tensor *slice = tensor_slice(T, 3, 7);
    
    assert(slice != NULL);
    assert(slice->shape[0] == 4);
    ASSERT_FLOAT_EQ(slice->data[0], 3.0f);
    ASSERT_FLOAT_EQ(slice->data[3], 6.0f);
    
    tensor_free(slice);
    tensor_free(T);
}

TEST(tensor_slice_invalid_range) {
    size_t shape[] = {5};
    Tensor *T = tensor_ones(shape, 1);
    
    Tensor *slice = tensor_slice(T, 3, 2); // start > end
    assert(slice == NULL);
    
    slice = tensor_slice(T, 0, 10); // end > size
    assert(slice == NULL);
    
    tensor_free(T);
}

// ============ NULL and Edge Case Tests ============

TEST(ops_null_inputs) {
    size_t shape[] = {2, 2};
    Tensor *T = tensor_ones(shape, 2);
    
    assert(tensor_add(NULL, T) == NULL);
    assert(tensor_add(T, NULL) == NULL);
    assert(tensor_sub(NULL, T) == NULL);
    assert(tensor_mul(NULL, T) == NULL);
    assert(tensor_matmul(NULL, T) == NULL);
    assert(tensor_transpose(NULL) == NULL);
    assert(tensor_relu(NULL) == NULL);
    assert(tensor_sigmoid(NULL) == NULL);
    assert(tensor_tanh(NULL) == NULL);
    assert(tensor_softmax(NULL) == NULL);
    
    tensor_free(T);
}

TEST(loss_incompatible_shapes) {
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 2};
    
    Tensor *A = tensor_ones(shape_a, 2);
    Tensor *B = tensor_ones(shape_b, 2);
    
    assert(tensor_mse(A, B) == NULL);
    assert(tensor_cross_entropy(A, B) == NULL);
    assert(tensor_binary_cross_entropy(A, B) == NULL);
    
    tensor_free(A);
    tensor_free(B);
}

int main() {
    printf("\n=== Running Ops Unit Tests ===\n\n");
    
    // Elementwise operations
    RUN_TEST(tensor_add_basic);
    RUN_TEST(tensor_add_broadcasting);
    RUN_TEST(tensor_add_autograd);
    RUN_TEST(tensor_sub_basic);
    RUN_TEST(tensor_mul_basic);
    
    // Matrix multiplication
    RUN_TEST(tensor_matmul_vector_vector);
    RUN_TEST(tensor_matmul_matrix_vector);
    RUN_TEST(tensor_matmul_vector_matrix);
    RUN_TEST(tensor_matmul_matrix_matrix);
    RUN_TEST(tensor_matmul_incompatible_shapes);
    RUN_TEST(tensor_matmul_identity);
    
    // Transpose
    RUN_TEST(tensor_transpose_basic);
    RUN_TEST(tensor_transpose_square);
    RUN_TEST(tensor_transpose_double);
    
    // Activation functions
    RUN_TEST(tensor_relu_basic);
    RUN_TEST(tensor_relu_all_positive);
    RUN_TEST(tensor_sigmoid_basic);
    RUN_TEST(tensor_sigmoid_range);
    RUN_TEST(tensor_tanh_basic);
    RUN_TEST(tensor_tanh_range);
    RUN_TEST(tensor_softmax_basic);
    RUN_TEST(tensor_softmax_numerical_stability);
    
    // Loss functions
    RUN_TEST(tensor_mse_basic);
    RUN_TEST(tensor_mse_nonzero);
    RUN_TEST(tensor_cross_entropy_basic);
    RUN_TEST(tensor_binary_cross_entropy_basic);
    
    // Tensor slice
    RUN_TEST(tensor_slice_basic);
    RUN_TEST(tensor_slice_1d);
    RUN_TEST(tensor_slice_invalid_range);
    
    // Edge cases
    RUN_TEST(ops_null_inputs);
    RUN_TEST(loss_incompatible_shapes);
    
    printf("\n=== All Ops Tests Passed! ===\n\n");
    return 0;
}
