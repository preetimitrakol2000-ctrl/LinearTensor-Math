#include <stdio.h>
#include "tensor.h"

int main() {
    printf("🧠 Activating LinearTensor-Math Flattened Activation Layer...\n\n");

    // Constructing a 1x3 Input Row Vector (e.g., Hidden Layer Activations)
    FlatTensor inputs = construct_tensor(1, 3);
    inputs.storage_vector[0] = 0.5; inputs.storage_vector[1] = -1.2; inputs.storage_vector[2] = 2.0;

    // Constructing a 3x1 Output Connection Weight Column Matrix
    FlatTensor weights = construct_tensor(3, 1);
    weights.storage_vector[0] = 0.8; weights.storage_vector[1] = 0.4; weights.storage_vector[2] = -0.5;

    printf("📥 Input Flat Representation: [ 0.5, -1.2, 2.0 ]\n");

    // Execute forward path dot calculation steps
    FlatTensor output_node = execute_dot_product(&inputs, &weights);
    printf("💥 Pre-Activation Node Inner Product Result: %.4f\n", output_node.storage_vector[0]);

    compute_sigmoid(&output_node);
    printf("✨ Post-Sigmoid Tensor Probability Node Output: %.4f\n", output_node.storage_vector[0]);

    free_tensor(&inputs);
    free_tensor(&weights);
    free_tensor(&output_node);
    return 0;
}
