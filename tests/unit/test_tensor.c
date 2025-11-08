/*
 * AI Generated Unit tests for tensor.c
 * Tests tensor creation, initialization, autograd, memory management, and utilities
 */

#include "lessbasednn/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define EPSILON 1e-6
#define ASSERT_FLOAT_EQ(a, b) assert(fabsf((a) - (b)) < EPSILON)
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    printf("Running test_%s...\n", #name); \
    test_##name(); \
    printf("  ✓ test_%s passed\n", #name); \
} while(0)

// Helper function to compare tensors
static int tensors_equal(Tensor *A, Tensor *B, float eps) {
    if (!A || !B) return 0;
    if (A->ndim != B->ndim) return 0;
    if (A->size != B->size) return 0;
    
    for (size_t i = 0; i < A->ndim; i++) {
        if (A->shape[i] != B->shape[i]) return 0;
    }
    
    for (size_t i = 0; i < A->size; i++) {
        if (fabsf(A->data[i] - B->data[i]) > eps) return 0;
    }
    
    return 1;
}

// Test tensor_create basic functionality
TEST(tensor_create_basic) {
    size_t shape[] = {3, 4};
    Tensor *T = tensor_create(shape, 2);
    
    assert(T != NULL);
    assert(T->ndim == 2);
    assert(T->shape[0] == 3);
    assert(T->shape[1] == 4);
    assert(T->size == 12);
    assert(T->data != NULL);
    assert(T->grad == NULL);
    assert(T->requires_grad == 0);
    assert(T->owns_data == 1);
    assert(T->op == OP_NONE);
    assert(T->inputs == NULL);
    assert(T->num_inputs == 0);
    
    tensor_free(T);
}

// Test 1D tensor creation
TEST(tensor_create_1d) {
    size_t shape[] = {10};
    Tensor *T = tensor_create(shape, 1);
    
    assert(T != NULL);
    assert(T->ndim == 1);
    assert(T->shape[0] == 10);
    assert(T->size == 10);
    
    tensor_free(T);
}

// Test 3D tensor creation
TEST(tensor_create_3d) {
    size_t shape[] = {2, 3, 4};
    Tensor *T = tensor_create(shape, 3);
    
    assert(T != NULL);
    assert(T->ndim == 3);
    assert(T->shape[0] == 2);
    assert(T->shape[1] == 3);
    assert(T->shape[2] == 4);
    assert(T->size == 24);
    
    tensor_free(T);
}

// Test tensor_zeroes
TEST(tensor_zeroes_basic) {
    size_t shape[] = {3, 3};
    Tensor *T = tensor_zeroes(shape, 2);
    
    assert(T != NULL);
    for (size_t i = 0; i < T->size; i++) {
        ASSERT_FLOAT_EQ(T->data[i], 0.0f);
    }
    
    tensor_free(T);
}

// Test tensor_ones
TEST(tensor_ones_basic) {
    size_t shape[] = {2, 4};
    Tensor *T = tensor_ones(shape, 2);
    
    assert(T != NULL);
    for (size_t i = 0; i < T->size; i++) {
        ASSERT_FLOAT_EQ(T->data[i], 1.0f);
    }
    
    tensor_free(T);
}

// Test tensor_randn with different seeds
TEST(tensor_randn_basic) {
    size_t shape[] = {100};
    Tensor *T = tensor_randn(shape, 1, 42);
    
    assert(T != NULL);
    
    // Check that values follow a distribution (not all zeros or ones)
    int has_negative = 0;
    int has_positive = 0;
    float sum = 0.0f;
    
    for (size_t i = 0; i < T->size; i++) {
        sum += T->data[i];
        if (T->data[i] < 0) has_negative = 1;
        if (T->data[i] > 0) has_positive = 1;
    }
    
    assert(has_negative && has_positive);
    
    // Mean should be close to 0 for large sample
    float mean = sum / T->size;
    assert(fabsf(mean) < 0.3f);
    
    tensor_free(T);
}

// Test tensor_randn reproducibility with same seed
TEST(tensor_randn_reproducibility) {
    size_t shape[] = {5, 5};
    Tensor *T1 = tensor_randn(shape, 2, 123);
    Tensor *T2 = tensor_randn(shape, 2, 123);
    
    assert(tensors_equal(T1, T2, EPSILON));
    
    tensor_free(T1);
    tensor_free(T2);
}

// Test tensor_randn different seeds produce different results
TEST(tensor_randn_different_seeds) {
    size_t shape[] = {5, 5};
    Tensor *T1 = tensor_randn(shape, 2, 42);
    Tensor *T2 = tensor_randn(shape, 2, 99);
    
    assert(!tensors_equal(T1, T2, EPSILON));
    
    tensor_free(T1);
    tensor_free(T2);
}

// Test tensor_free with NULL
TEST(tensor_free_null) {
    tensor_free(NULL);  // Should not crash
}

// Test tensor_set_requires_grad
TEST(tensor_set_requires_grad_basic) {
    size_t shape[] = {3, 3};
    Tensor *T = tensor_create(shape, 2);
    
    assert(T->requires_grad == 0);
    
    tensor_set_requires_grad(T, 1);
    assert(T->requires_grad == 1);
    
    tensor_set_requires_grad(T, 0);
    assert(T->requires_grad == 0);
    
    tensor_free(T);
}

// Test tensor_zero_grad
TEST(tensor_zero_grad_basic) {
    size_t shape[] = {2, 3};
    Tensor *T = tensor_create(shape, 2);
    
    // Allocate and fill gradient
    T->grad = (float *)malloc(T->size * sizeof(float));
    for (size_t i = 0; i < T->size; i++) {
        T->grad[i] = (float)i;
    }
    
    tensor_zero_grad(T);
    
    for (size_t i = 0; i < T->size; i++) {
        ASSERT_FLOAT_EQ(T->grad[i], 0.0f);
    }
    
    tensor_free(T);
}

// Test tensor_zero_grad with NULL gradient
TEST(tensor_zero_grad_null) {
    size_t shape[] = {2, 3};
    Tensor *T = tensor_create(shape, 2);
    
    tensor_zero_grad(T);  // Should not crash
    
    tensor_free(T);
}

// Test tensor_fill
TEST(tensor_fill_basic) {
    size_t shape[] = {3, 4};
    Tensor *T = tensor_create(shape, 2);
    
    tensor_fill(T, 5.5f);
    
    for (size_t i = 0; i < T->size; i++) {
        ASSERT_FLOAT_EQ(T->data[i], 5.5f);
    }
    
    tensor_free(T);
}

// Test tensor_fill with negative value
TEST(tensor_fill_negative) {
    size_t shape[] = {2, 2};
    Tensor *T = tensor_create(shape, 2);
    
    tensor_fill(T, -3.14f);
    
    for (size_t i = 0; i < T->size; i++) {
        ASSERT_FLOAT_EQ(T->data[i], -3.14f);
    }
    
    tensor_free(T);
}

// Test tensor_copy
TEST(tensor_copy_basic) {
    size_t shape[] = {3, 3};
    Tensor *T = tensor_randn(shape, 2, 42);
    
    Tensor *C = tensor_copy(T);
    
    assert(C != NULL);
    assert(tensors_equal(T, C, EPSILON));
    assert(C != T);  // Different objects
    assert(C->data != T->data);  // Different data arrays
    
    tensor_free(T);
    tensor_free(C);
}

// Test tensor_copy preserves shape
TEST(tensor_copy_preserves_shape) {
    size_t shape[] = {2, 3, 4};
    Tensor *T = tensor_ones(shape, 3);
    
    Tensor *C = tensor_copy(T);
    
    assert(C->ndim == T->ndim);
    for (size_t i = 0; i < T->ndim; i++) {
        assert(C->shape[i] == T->shape[i]);
    }
    assert(C->size == T->size);
    
    tensor_free(T);
    tensor_free(C);
}

// Test tensor_copy independence
TEST(tensor_copy_independence) {
    size_t shape[] = {2, 2};
    Tensor *T = tensor_ones(shape, 2);
    Tensor *C = tensor_copy(T);
    
    // Modify original
    T->data[0] = 99.0f;
    
    // Copy should be unchanged
    ASSERT_FLOAT_EQ(C->data[0], 1.0f);
    
    tensor_free(T);
    tensor_free(C);
}

// Test tensor_backward basic scalar
TEST(tensor_backward_scalar) {
    size_t shape[] = {1};
    Tensor *T = tensor_create(shape, 1);
    T->data[0] = 5.0f;
    
    tensor_set_requires_grad(T, 1);
    tensor_backward(T);
    
    assert(T->grad != NULL);
    ASSERT_FLOAT_EQ(T->grad[0], 1.0f);
    
    tensor_free(T);
}

// Test tensor_backward with no requires_grad
TEST(tensor_backward_no_grad) {
    size_t shape[] = {2, 2};
    Tensor *T = tensor_create(shape, 2);
    T->requires_grad = 0;
    
    tensor_backward(T);
    
    // Should not crash and grad should remain NULL
    assert(T->grad == NULL);
    
    tensor_free(T);
}

// Test memory management - multiple allocations
TEST(memory_stress_test) {
    const int num_tensors = 100;
    Tensor *tensors[num_tensors];
    
    // Allocate many tensors
    for (int i = 0; i < num_tensors; i++) {
        size_t shape[] = {10, 10};
        tensors[i] = tensor_randn(shape, 2, i);
        assert(tensors[i] != NULL);
    }
    
    // Free all tensors
    for (int i = 0; i < num_tensors; i++) {
        tensor_free(tensors[i]);
    }
}

// Test large tensor
TEST(tensor_large) {
    size_t shape[] = {100, 100};
    Tensor *T = tensor_zeroes(shape, 2);
    
    assert(T != NULL);
    assert(T->size == 10000);
    
    tensor_free(T);
}

// Test edge case: size 1 tensor
TEST(tensor_size_one) {
    size_t shape[] = {1, 1, 1};
    Tensor *T = tensor_create(shape, 3);
    
    assert(T != NULL);
    assert(T->size == 1);
    assert(T->ndim == 3);
    
    T->data[0] = 42.0f;
    ASSERT_FLOAT_EQ(T->data[0], 42.0f);
    
    tensor_free(T);
}

// Test tensor with zero dimension (degenerate case)
TEST(tensor_zero_dimension) {
    size_t shape[] = {5, 0, 3};
    Tensor *T = tensor_create(shape, 3);
    
    assert(T != NULL);
    assert(T->size == 0);  // 5 * 0 * 3 = 0
    
    tensor_free(T);
}

// Test tensor_copy with NULL
TEST(tensor_copy_null) {
    Tensor *C = tensor_copy(NULL);
    assert(C == NULL);
}

// Test requires_grad propagation
TEST(requires_grad_no_propagation) {
    size_t shape[] = {2, 2};
    Tensor *T = tensor_randn(shape, 2, 42);
    
    tensor_set_requires_grad(T, 1);
    
    Tensor *C = tensor_copy(T);
    
    // Copy should not have requires_grad set
    assert(C->requires_grad == 0);
    assert(C->grad == NULL);
    assert(C->inputs == NULL);
    
    tensor_free(T);
    tensor_free(C);
}

// Test OpType enum values
TEST(op_type_values) {
    size_t shape[] = {1};
    Tensor *T = tensor_create(shape, 1);
    
    assert(T->op == OP_NONE);
    
    T->op = OP_ADD;
    assert(T->op == OP_ADD);
    
    T->op = OP_MATMUL;
    assert(T->op == OP_MATMUL);
    
    tensor_free(T);
}

// Test tensor data modification
TEST(tensor_data_modification) {
    size_t shape[] = {3, 3};
    Tensor *T = tensor_zeroes(shape, 2);
    
    // Modify data
    for (size_t i = 0; i < T->size; i++) {
        T->data[i] = (float)(i * 2);
    }
    
    // Verify modification
    for (size_t i = 0; i < T->size; i++) {
        ASSERT_FLOAT_EQ(T->data[i], (float)(i * 2));
    }
    
    tensor_free(T);
}

// Test gradient allocation during backward
TEST(gradient_allocation_backward) {
    size_t shape[] = {3, 3};
    Tensor *T = tensor_randn(shape, 2, 42);
    
    tensor_set_requires_grad(T, 1);
    
    assert(T->grad == NULL);  // No gradient initially
    
    tensor_backward(T);
    
    assert(T->grad != NULL);  // Gradient allocated during backward
    
    // All gradients should be 1.0 for scalar output
    for (size_t i = 0; i < T->size; i++) {
        ASSERT_FLOAT_EQ(T->grad[i], 1.0f);
    }
    
    tensor_free(T);
}

int main() {
    printf("\n=== Running Tensor Unit Tests ===\n\n");
    
    RUN_TEST(tensor_create_basic);
    RUN_TEST(tensor_create_1d);
    RUN_TEST(tensor_create_3d);
    RUN_TEST(tensor_zeroes_basic);
    RUN_TEST(tensor_ones_basic);
    RUN_TEST(tensor_randn_basic);
    RUN_TEST(tensor_randn_reproducibility);
    RUN_TEST(tensor_randn_different_seeds);
    RUN_TEST(tensor_free_null);
    RUN_TEST(tensor_set_requires_grad_basic);
    RUN_TEST(tensor_zero_grad_basic);
    RUN_TEST(tensor_zero_grad_null);
    RUN_TEST(tensor_fill_basic);
    RUN_TEST(tensor_fill_negative);
    RUN_TEST(tensor_copy_basic);
    RUN_TEST(tensor_copy_preserves_shape);
    RUN_TEST(tensor_copy_independence);
    RUN_TEST(tensor_backward_scalar);
    RUN_TEST(tensor_backward_no_grad);
    RUN_TEST(memory_stress_test);
    RUN_TEST(tensor_large);
    RUN_TEST(tensor_size_one);
    RUN_TEST(tensor_zero_dimension);
    RUN_TEST(tensor_copy_null);
    RUN_TEST(requires_grad_no_propagation);
    RUN_TEST(op_type_values);
    RUN_TEST(tensor_data_modification);
    RUN_TEST(gradient_allocation_backward);
    
    printf("\n=== All Tensor Tests Passed! ===\n\n");
    return 0;
}
