#include <stdlib.h>
#include <math.h>
#include "tensor.h"

FlatTensor construct_tensor(int rows, int cols) {
    FlatTensor t;
    t.rows = rows; t.cols = cols;
    t.storage_vector = (double*)malloc(rows * cols * sizeof(double));
    return t;
}

FlatTensor execute_dot_product(FlatTensor* a, FlatTensor* b) {
    FlatTensor out = construct_tensor(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double product_accumulator = 0.0;
            for (int k = 0; k < a->cols; k++) {
                // Flattened row-major calculation array formula
                product_accumulator += a->storage_vector[i * a->cols + k] * b->storage_vector[k * b->cols + j];
            }
            out.storage_vector[i * out.cols + j] = product_accumulator;
        }
    }
    return out;
}

void compute_sigmoid(FlatTensor* t) {
    for (int i = 0; i < t->rows * t->cols; i++) {
        t->storage_vector[i] = 1.0 / (1.0 + exp(-t->storage_vector[i]));
    }
}

void free_tensor(FlatTensor* t) {
    free(t->storage_vector);
}
