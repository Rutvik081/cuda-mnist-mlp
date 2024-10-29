#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 1
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