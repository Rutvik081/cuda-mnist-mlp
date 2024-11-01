#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 4
#define EPOCHS 10
#define LEARNING_RATE 0.001

typedef struct {
    float *w1;          // layer 1 weights
    float *w2;          // layer 2 weights
    float *b1;          // layer 1 bias
    float *b2;          // layer 2 bias
    float *grad_w1;     // w1 gradients
    float *grad_w2;     // w2 gradients
    float *grad_b1;     // b1 gradients
    float *grad_b2;     // b2 gradients
} NeuralNetwork;

// load batched data
void load_data(const char *filename, float *data, int size){
    FILE *file = fopen(filename, "rb");
    if (file == NULL){
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size){
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// load batch labels
void load_labels(const char *filename, int *labels, int size){
    FILE *file = fopen(filename, "rb");
    if (file == NULL){
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size){
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// Kaiming initialization for weights in neural networks
void weights_init(float *weights, int size){
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++){
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

// Initializing bias
void bias_init(float *bias, int size){
    for (int i = 0; i < size; i++){
        bias[i] = 0.0f;
    }
}

// Initializing the neural network
void neural_network_init(NeuralNetwork *nn){
    nn->w1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn->w2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->b1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->b2 = malloc(OUTPUT_SIZE * sizeof(float));
    nn->grad_w1 = malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    nn->grad_w2 = malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->grad_b1 = malloc(HIDDEN_SIZE * sizeof(float));
    nn->grad_b2 = malloc(OUTPUT_SIZE * sizeof(float));

    weights_init(nn->w1, HIDDEN_SIZE * INPUT_SIZE);
    weights_init(nn->w2, OUTPUT_SIZE * HIDDEN_SIZE);
    bias_init(nn->b1, HIDDEN_SIZE);
    bias_init(nn->b2, OUTPUT_SIZE);
}

void matmul_a_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

// Add bias
void bias_forward(float *x, float *bias, int batch_size, int size){
    for (int b = 0; b < batch_size; b++){
        for (int i = 0; i < size; i++){
            x[b * size + i] += bias[i];
        }
    }
}

// ReLU forward
void relu_forward(float *x, int size){
    for (int i = 0; i < size; i++){
        x[i] = fmaxf(0.0f, x[i]);
    }
}

void softmax(float *x, int batch_size, int size){
    for (int b = 0; b < batch_size; b++){
        float max = x[b * size];
        for (int i = 1; i < size; i++){
            if (x[b * size + i] > max) max = x[b * size + i];
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++){
            x[b * size + i] = expf(x[b * size + i] - max);
            sum += x[b * size + i];
        }
        for (int i = 0; i < size; i++){
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

// Forward pass function
void forward(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size){
    
    // Input to Hidden (X @ W1)
    matmul_a_b(input, nn->w1, hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);

    // Add b1
    bias_forward(hidden, nn->b1, batch_size, HIDDEN_SIZE);

    // Apply ReLU
    relu_forward(hidden, batch_size * HIDDEN_SIZE);

    // Hidden to Output (Hidden @ W2)
    matmul_a_b(hidden, nn->w2, output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);

    // Add b2
    bias_forward(output, nn->b2, batch_size, OUTPUT_SIZE);

    // Apply softmax
    softmax(output, batch_size, OUTPUT_SIZE);
}

float cross_entropy_loss(float *output, int *labels, int batch_size){
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++){
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}

// Zero out gradients
void zero_grad(float *grad, int size){
    memset(grad, 0, size * sizeof(float));
}

// Compute gradients for output layer
void compute_output_gradients(float *grad_output, float *output, int *labels, int batch_size){
    for (int b = 0; b < batch_size; b++){
        for (int i = 0; i < OUTPUT_SIZE; i++){
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
}

// Matrix Multiplication A.T @ B
void matmul_at_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < m; l++) {
                C[i * k + j] += A[l * n + i] * B[l * k + j];
            }
        }
    }
}

// Bias backward
void bias_backward(float *grad_bias, float *grad, int batch_size, int size){
    for (int i = 0; i < size; i++){
        grad_bias[i] = 0.0f;
        for (int b = 0; b < batch_size; b++){
            grad_bias[i] += grad[b * size + i];
        }
    }
}

// Matrix Multiplication A @ B.T
void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[j * n + l];
            }
        }
    }
}

// Backward pass function
void backward(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size){

    // Inititalize gradients to zero
    zero_grad(nn->grad_w1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad(nn->grad_w2, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad(nn->grad_b1, HIDDEN_SIZE);
    zero_grad(nn->grad_b2, OUTPUT_SIZE);

    // Compute gradients for output layer
    float *grad_output = malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    compute_output_gradients(grad_output, output, labels, batch_size);

    // Update gradients for w2 (W2.grad = grad_output.T @ hidden)
    matmul_at_b(hidden, grad_output, nn->grad_w2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);

    // Update gradients for b2
    bias_backward(nn->grad_b2, grad_output, batch_size, OUTPUT_SIZE);

    // Compute dX2 (gradient of loss w.r.t. input of second layer)
    float *dX2 = malloc(batch_size * HIDDEN_SIZE * sizeof(float));

    // grad_output @ W2.T = dX2 -> (B, 10) @ (10, 256) = (B, 256)
    matmul_a_bt(grad_output, nn->w2, dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);

    // Compute d_ReLU_out (element-wise multiplication with ReLU derivative)
    float *d_ReLU_out = malloc(batch_size * HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < batch_size * HIDDEN_SIZE; i++){
        d_ReLU_out[i] = dX2[i] * (hidden[i] > 0);
    }

    // retains its shape since its just a point-wise operation
    // Update gradients for w1 (W1.grad = d_ReLU_out.T @ input)
    matmul_at_b(input, d_ReLU_out, nn->grad_w1, batch_size, INPUT_SIZE, HIDDEN_SIZE);

    // Update gradients for b1
    bias_backward(nn->grad_b1, d_ReLU_out, batch_size, HIDDEN_SIZE);

    // Free allocated memory
    free(grad_output);
    free(dX2);
    free(d_ReLU_out);

}

// gradient descent step
void update_weights(NeuralNetwork *nn){
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++){
        nn->w1[i] -= LEARNING_RATE * nn->grad_w1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++){
        nn->w2[i] -= LEARNING_RATE * nn->grad_w2[i];
    }
    for (int i = 0; i < HIDDEN_SIZE; i++){
        nn->b1[i] -= LEARNING_RATE * nn->grad_b1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++){
        nn->b2[i] -= LEARNING_RATE * nn->grad_b2[i];
    }
}

void train(NeuralNetwork *nn, float *X_train, int *y_train){
    float *hidden = malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));

    int num_batches = TRAIN_SIZE / BATCH_SIZE;

    for (int epoch = 0; epoch < EPOCHS; epoch++){
        float total_loss = 0.0f;
        int correct = 0;

        for (int batch = 0; batch < num_batches; batch++){
            int start_idx = batch * BATCH_SIZE;

            forward(nn, &X_train[start_idx * INPUT_SIZE], hidden, output, BATCH_SIZE);

            float loss = cross_entropy_loss(output, &y_train[start_idx], BATCH_SIZE);
            total_loss += loss;

            for (int i = 0; i < BATCH_SIZE; i++){
                int predicted = 0;
                for (int j = 1; j < OUTPUT_SIZE; j++){
                    if (output[i * OUTPUT_SIZE + j] > output[i * OUTPUT_SIZE + predicted]) predicted = j;
                }
                if (predicted == y_train[start_idx + i]) correct++;
            }

            backward(nn, &X_train[start_idx * INPUT_SIZE], hidden, output, &y_train[start_idx], BATCH_SIZE);
            update_weights(nn);

            if ((batch + 1) % 100 == 0 || (epoch == 0 && batch == 0)){
                printf("Epoch %d/%d, Iter %d/%d, Loss: %.4f, Accuracy: %.2f%%\n",
                        epoch + 1, EPOCHS, batch + 1, num_batches, total_loss / (batch + 1),
                        100.0f * correct / ((batch + 1) * BATCH_SIZE));
            }
        }

        printf("Epoch %d/%d completed, Loss: %.4f, Accuracy: %.2f%%\n",
                epoch + 1, EPOCHS, total_loss / num_batches, 100.0f * correct / TRAIN_SIZE);
    }

    free(hidden);
    free(output);
}

int main(){
    srand(time(NULL));

    NeuralNetwork nn;
    neural_network_init(&nn);

    float *X_train = malloc(TRAIN_SIZE * INPUT_SIZE * sizeof(float));
    int *y_train = malloc(TRAIN_SIZE * sizeof(int));
    float *X_test = malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *y_test = malloc(TEST_SIZE * sizeof(int));

    load_data("dataset/X_train.bin", X_train, TRAIN_SIZE * INPUT_SIZE);
    load_labels("dataset/y_train.bin", y_train, TRAIN_SIZE);
    load_data("dataset/X_test.bin", X_test, TEST_SIZE * INPUT_SIZE);
    load_labels("dataset/y_test.bin", y_test, TEST_SIZE);

    // print first image
    for (int i = 0; i < 28; i++){
        for (int j = 0; j < 28; j++){
            if(X_train[0 * INPUT_SIZE + i * 28 + j] > 0.0f) printf("X");
            else printf(" ");
        }
        printf("\n");
    }

    printf("First 10 training labels: ");
    for (int i = 0; i < 10; i++){
        printf("%d ", y_train[i]);
    }
    printf("\n");

    train(&nn, X_train, y_train);

    free(nn.w1);
    free(nn.w2);
    free(nn.b1);
    free(nn.b2);
    free(nn.grad_w1);
    free(nn.grad_w2);
    free(nn.grad_b1);
    free(nn.grad_b2);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}