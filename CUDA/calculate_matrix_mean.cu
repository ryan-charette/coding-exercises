#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>

__global__ void row_mean_kernel(
    const float* matrix,
    float* result,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;

        for (int col = 0; col < cols; col++) {
            sum += matrix[row * cols + col];
        }

        result[row] = sum / cols;
    }
}

__global__ void col_mean_kernel(
    const float* matrix,
    float* result,
    int rows,
    int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols) {
        float sum = 0.0f;

        for (int row = 0; row < rows; row++) {
            sum += matrix[row * cols + col];
        }

        result[col] = sum / rows;
    }
}

std::vector<float> calculate_matrix_mean(const std::vector<std::vector<float>>& matrix, const std::string& mode) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    std::vector<float> flat_matrix(rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat_matrix[i * cols + j] = matrix[i][j];
        }
    }

    int result_size;
    if (mode == "row") {
        result_size = rows;
    } else if (mode == "column") {
        result_size = cols;
    } else {
        throw std::runtime_error("mode must be 'row' or 'column'");
    }

    float* d_matrix = nullptr;
    float* d_result = nullptr;

    // Allocate device memory
    cudaMalloc((void**)&d_matrix, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_result, result_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_matrix, flat_matrix.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch appropriate kernel based on mode ('row' or 'column')
    int threadsPerBlock = 256;
    int blocksPerGrid;

    if (mode == "row") {
        blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
        row_mean_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_result, rows, cols);
    } else {
        blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;
        col_mean_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, d_result, rows, cols);
    }

    // Copy result back
    std::vector<float> result(result_size);
    cudaMemcpy(result.data(), d_result, result_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free memory and return result
    cudaFree(d_matrix);
    cudaFree(d_result);

    return result;
}
