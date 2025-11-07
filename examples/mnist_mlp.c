// AI MNIST MLP Training Example - PyTorch-style API
// Demonstrates a complete training loop for MNIST digit classification
// using a multi-layer perceptron with the Petard neural network library

#include "petard/network.h"
#include "petard/layer.h"
#include "petard/loss.h"
#include "petard/optimizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// MNIST dimensions
#define MNIST_IMAGE_SIZE 784  // 28x28 pixels flattened
#define MNIST_NUM_CLASSES 10  // Digits 0-9
#define BATCH_SIZE 32
#define NUM_EPOCHS 10
#define LEARNING_RATE 0.001f

// Helper function to load MNIST data (simplified - you'd load from actual files)
// In practice, you'd read from idx-ubyte format files
void load_mnist_batch(Tensor *images, Tensor *labels, int batch_idx) {
    // For this example, we'll generate random data
    // In a real implementation, you'd load from MNIST dataset files
    
    for (size_t i = 0; i < images->size; i++) {
        images->data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random [-1, 1]
    }
    
    // Generate random one-hot labels
    memset(labels->data, 0, labels->size * sizeof(float));
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        int class = rand() % MNIST_NUM_CLASSES;
        labels->data[i * MNIST_NUM_CLASSES + class] = 1.0f;
    }
}

// Calculate accuracy
float calculate_accuracy(Tensor *predictions, Tensor *targets) {
    int correct = 0;
    
    for (size_t i = 0; i < BATCH_SIZE; i++) {
        // Find predicted class (argmax)
        int pred_class = 0;
        float max_pred = predictions->data[i * MNIST_NUM_CLASSES];
        for (int j = 1; j < MNIST_NUM_CLASSES; j++) {
            if (predictions->data[i * MNIST_NUM_CLASSES + j] > max_pred) {
                max_pred = predictions->data[i * MNIST_NUM_CLASSES + j];
                pred_class = j;
            }
        }
        
        // Find true class (argmax of one-hot)
        int true_class = 0;
        for (int j = 0; j < MNIST_NUM_CLASSES; j++) {
            if (targets->data[i * MNIST_NUM_CLASSES + j] > 0.5f) {
                true_class = j;
                break;
            }
        }
        
        if (pred_class == true_class) {
            correct++;
        }
    }
    
    return (float)correct / BATCH_SIZE * 100.0f;
}

int main() {
    printf("=== MNIST MLP Training with Petard ===\n\n");
    
    srand(time(NULL));
    
    // =========================================================================
    // 1. Build the Network (PyTorch Sequential-style)
    // =========================================================================
    printf("Building MLP network...\n");
    Network *net = network_create();
    
    // Input layer: 784 -> 512 with ReLU
    network_add_layer(net, layer_linear_create(MNIST_IMAGE_SIZE, 512));
    network_add_layer(net, layer_relu_create());
    
    // Hidden layer 1: 512 -> 256 with ReLU
    network_add_layer(net, layer_linear_create(512, 256));
    network_add_layer(net, layer_relu_create());
    
    // Hidden layer 2: 256 -> 128 with ReLU
    network_add_layer(net, layer_linear_create(256, 128));
    network_add_layer(net, layer_relu_create());
    
    // Output layer: 128 -> 10 with Softmax
    network_add_layer(net, layer_linear_create(128, MNIST_NUM_CLASSES));
    network_add_layer(net, layer_softmax_create());
    
    printf("Network architecture:\n");
    printf("  Input:    784 (28x28 images)\n");
    printf("  Hidden 1: 512 neurons (ReLU)\n");
    printf("  Hidden 2: 256 neurons (ReLU)\n");
    printf("  Hidden 3: 128 neurons (ReLU)\n");
    printf("  Output:   10 classes (Softmax)\n");
    network_print(net);
    printf("\n");
    
    // =========================================================================
    // 2. Create Optimizer (Adam with default parameters)
    // =========================================================================
    printf("Creating Adam optimizer (lr=%.4f)...\n", LEARNING_RATE);
    Optimizer *optimizer = optimizer_adam_from_network(
        net,
        LEARNING_RATE,  // learning_rate
        0.9f,           // beta1
        0.999f,         // beta2
        1e-8f           // epsilon
    );
    printf("\n");
    
    // =========================================================================
    // 3. Training Loop
    // =========================================================================
    printf("Starting training...\n");
    printf("Epochs: %d, Batch size: %d\n\n", NUM_EPOCHS, BATCH_SIZE);
    
    size_t image_shape[] = {BATCH_SIZE, MNIST_IMAGE_SIZE};
    size_t label_shape[] = {BATCH_SIZE, MNIST_NUM_CLASSES};
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float epoch_loss = 0.0f;
        float epoch_accuracy = 0.0f;
        int num_batches = 100;  // Simulate 100 batches per epoch
        
        for (int batch = 0; batch < num_batches; batch++) {
            // -------------------------------------------------------------------
            // 3a. Load batch data
            // -------------------------------------------------------------------
            Tensor *images = tensor_create(image_shape, 2);
            Tensor *labels = tensor_create(label_shape, 2);
            load_mnist_batch(images, labels, batch);
            
            // Input doesn't need gradients
            tensor_set_requires_grad(images, 0);
            
            // -------------------------------------------------------------------
            // 3b. Forward pass
            // -------------------------------------------------------------------
            Tensor *predictions = network_forward(net, images);
            
            // -------------------------------------------------------------------
            // 3c. Compute loss (builds computation graph for autograd)
            // -------------------------------------------------------------------
            Tensor *loss_tensor = loss_cross_entropy(predictions, labels);
            float loss_value = loss_tensor->data[0];
            epoch_loss += loss_value;
            
            // Calculate accuracy
            float batch_accuracy = calculate_accuracy(predictions, labels);
            epoch_accuracy += batch_accuracy;
            
            // -------------------------------------------------------------------
            // 3d. Backward pass (autograd computes all gradients)
            // -------------------------------------------------------------------
            tensor_backward(loss_tensor);
            
            // -------------------------------------------------------------------
            // 3e. Update weights using optimizer
            // -------------------------------------------------------------------
            optimizer_step(optimizer);
            
            // -------------------------------------------------------------------
            // 3f. Zero gradients for next iteration
            // -------------------------------------------------------------------
            optimizer_zero_grad(optimizer);
            
            // -------------------------------------------------------------------
            // 3g. Cleanup batch tensors
            // -------------------------------------------------------------------
            tensor_free(images);
            tensor_free(labels);
            tensor_free(loss_tensor);
            // predictions is freed by the autograd system
            
            // Print progress every 20 batches
            if ((batch + 1) % 20 == 0) {
                printf("  Epoch %d/%d, Batch %d/%d - Loss: %.4f, Accuracy: %.2f%%\r",
                       epoch + 1, NUM_EPOCHS, batch + 1, num_batches,
                       loss_value, batch_accuracy);
                fflush(stdout);
            }
        }
        
        // Print epoch summary
        epoch_loss /= num_batches;
        epoch_accuracy /= num_batches;
        printf("\nEpoch %d/%d completed - Avg Loss: %.4f, Avg Accuracy: %.2f%%\n\n",
               epoch + 1, NUM_EPOCHS, epoch_loss, epoch_accuracy);
    }
    
    // =========================================================================
    // 4. Evaluation Mode (inference without computing gradients)
    // =========================================================================
    printf("\n=== Evaluation ===\n");
    printf("Running inference on test batch...\n");
    
    Tensor *test_images = tensor_create(image_shape, 2);
    Tensor *test_labels = tensor_create(label_shape, 2);
    load_mnist_batch(test_images, test_labels, 0);
    tensor_set_requires_grad(test_images, 0);
    
    // Forward pass without computing gradients
    Tensor *test_predictions = network_forward(net, test_images);
    
    // Use value-only loss function (no autograd)
    float test_loss = loss_cross_entropy_value(test_predictions, test_labels);
    float test_accuracy = calculate_accuracy(test_predictions, test_labels);
    
    printf("Test Loss: %.4f\n", test_loss);
    printf("Test Accuracy: %.2f%%\n", test_accuracy);
    
    // Print some predictions
    printf("\nSample predictions (first 5 images):\n");
    for (int i = 0; i < 5; i++) {
        int pred_class = 0;
        float max_prob = test_predictions->data[i * MNIST_NUM_CLASSES];
        for (int j = 1; j < MNIST_NUM_CLASSES; j++) {
            if (test_predictions->data[i * MNIST_NUM_CLASSES + j] > max_prob) {
                max_prob = test_predictions->data[i * MNIST_NUM_CLASSES + j];
                pred_class = j;
            }
        }
        
        int true_class = 0;
        for (int j = 0; j < MNIST_NUM_CLASSES; j++) {
            if (test_labels->data[i * MNIST_NUM_CLASSES + j] > 0.5f) {
                true_class = j;
                break;
            }
        }
        
        printf("  Image %d: Predicted=%d (%.2f%%), True=%d %s\n",
               i + 1, pred_class, max_prob * 100.0f, true_class,
               (pred_class == true_class) ? "✓" : "✗");
    }
    
    // =========================================================================
    // 5. Cleanup
    // =========================================================================
    printf("\nCleaning up...\n");
    tensor_free(test_images);
    tensor_free(test_labels);
    // test_predictions freed by autograd
    
    optimizer_free(optimizer);
    network_free(net);
    
    printf("Training complete!\n");
    
    return 0;
}
