#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    int rows;
    int cols;
    double* storage_vector;
} FlatTensor;

FlatTensor construct_tensor(int rows, int cols);
void free_tensor(FlatTensor* t);
FlatTensor execute_dot_product(FlatTensor* a, FlatTensor* b);
void compute_sigmoid(FlatTensor* t);

#endif
