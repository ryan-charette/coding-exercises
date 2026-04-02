#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrix_vector_dot_kernel(
    const float* matrix,
    const float* vector,
    float* result,
    int rows,
    int cols
) {
    // Each thread computes one row's dot product
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float dot = 0.0f;

        for (int i = 0; i < cols; i++) {
            dot += matrix[row * cols + i] * vector[i];
        }

        result[row] = dot;
    }
}

std::vector<float> matrix_dot_vector(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {    
    int rows = matrix.size();
    int cols = vec.size();

    std::vector<float> result(rows);

    float* d_matrix;
    float* d_vector;
    float* d_result;

    // Return empty vector if dimensions don't match
    if (matrix[0].size() != cols) {
        return {};
    }    

    // Flatten the 2D matrix into a 1D array
    std::vector<float> flat_matrix(rows * cols);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            flat_matrix[r * cols + c] = matrix[r][c];
        }
    }

    // Allocate device memory
    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    cudaMalloc(&d_vector, cols * sizeof(float));
    cudaMalloc(&d_result, rows * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_matrix, flat_matrix.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vec.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    matrix_vector_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_matrix, d_vector, d_result, rows, cols
    );

    // Copy result back
    cudaMemcpy(result.data(), d_result, rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory and return result
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return result;
}
