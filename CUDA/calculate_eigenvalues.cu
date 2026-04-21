#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void eigenvalue_kernel(
    const float* matrix,
    float* eigenvalues
) {
    float a = matrix[0];  // matrix[0][0]
    float b = matrix[1];  // matrix[0][1]
    float c = matrix[2];  // matrix[1][0]
    float d = matrix[3];  // matrix[1][1]
    
    float trace = a + d;
    float det = a * d - b * c;
    float discriminant = trace * trace - 4.0f * det;
    
    float sqrt_disc = sqrtf(discriminant);
    eigenvalues[0] = (trace + sqrt_disc) / 2.0f;
    eigenvalues[1] = (trace - sqrt_disc) / 2.0f;
}

std::vector<float> calculate_eigenvalues(const std::vector<std::vector<float>>& matrix) {
    // Flatten 2x2 matrix
    std::vector<float> flat_matrix = {
        matrix[0][0], matrix[0][1],
        matrix[1][0], matrix[1][1]
    };
    
    float *d_matrix, *d_eigenvalues;
    std::vector<float> h_eigenvalues(2);
    
    cudaMalloc(&d_matrix, 4 * sizeof(float));
    cudaMalloc(&d_eigenvalues, 2 * sizeof(float));
    
    cudaMemcpy(d_matrix, flat_matrix.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
    
    eigenvalue_kernel<<<1, 1>>>(d_matrix, d_eigenvalues);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_eigenvalues.data(), d_eigenvalues, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix);
    cudaFree(d_eigenvalues);
    
    return h_eigenvalues;
}
