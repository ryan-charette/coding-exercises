#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void transpose_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

std::vector<std::vector<float>> transpose_matrix(const std::vector<std::vector<float>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    std::vector<float> flat_output(cols * rows);

    float* d_input;
    float* d_output;

    // Flatten the 2D matrix into a 1D array
    std::vector<float> flat_input(rows * cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            flat_input[r * cols + c] = matrix[r][c];
        }
    }

    // Allocate device memory
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, cols * rows * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, flat_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                  (rows + blockSize.y - 1) / blockSize.y);
    
    transpose_kernel<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);

    // Copy result back
    cudaMemcpy(flat_output.data(), d_output, cols * rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory and return result
    cudaFree(d_input);
    cudaFree(d_output);

    std::vector<std::vector<float>> result(cols, std::vector<float>(rows));
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            result[c][r] = flat_output[c * rows + r];
        }
    }
    
    return result;
}
