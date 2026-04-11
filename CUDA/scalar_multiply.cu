#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void scalar_multiply_kernel(
    const float* input,
    float* output,
    float scalar,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        output[idx] = input[idx] * scalar;
    }
}

std::vector<std::vector<float>> scalar_multiply(const std::vector<std::vector<float>>& matrix, float scalar) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    int total_elements = rows * cols;

    std::vector<float> flat_input(rows * cols);
    std::vector<float> flat_output(total_elements);

    float* d_input = nullptr;
    float* d_output = nullptr;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            flat_input[r * cols + c] = matrix[r][c];
        }
    }
    
    // Allocate device memory
    cudaMalloc(&d_input, total_elements * sizeof(float));
    cudaMalloc(&d_output, total_elements * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, flat_input.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;

    scalar_multiply_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_input, d_output, scalar, total_elements
    );

    // Copy result back
    cudaMemcpy(flat_output.data(), d_output, total_elements * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory and return result
    cudaFree(d_input);
    cudaFree(d_output);

    std::vector<std::vector<float>> result(rows, std::vector<float>(cols));

    int i = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            result[r][c] = flat_output[i];
            i++;
        }
    }

    return result;
}
