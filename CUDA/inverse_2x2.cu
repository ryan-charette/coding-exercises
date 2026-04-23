#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void inverse_2x2_kernel(
    const float* matrix,
    float* inverse,
    int* is_invertible
) {
    float a = matrix[0];
    float b = matrix[1];
    float c = matrix[2];
    float d = matrix[3];
    
    float det = a * d - b * c;
    
    if (det == 0.0f) {
        *is_invertible = 0;
        return;
    }
    
    *is_invertible = 1;
    
    inverse[0] =  d / det;
    inverse[1] = -b / det;
    inverse[2] = -c / det;
    inverse[3] =  a / det;
}

std::vector<std::vector<float>> inverse_2x2(const std::vector<std::vector<float>>& matrix) {
    std::vector<float> flat_matrix = {
        matrix[0][0], matrix[0][1],
        matrix[1][0], matrix[1][1]
    };
    
    float *d_matrix, *d_inverse;
    int *d_invertible;
    
    std::vector<float> h_inverse(4);
    int h_invertible = 0;
    
    cudaMalloc(&d_matrix, 4 * sizeof(float));
    cudaMalloc(&d_inverse, 4 * sizeof(float));
    cudaMalloc(&d_invertible, sizeof(int));
    
    cudaMemcpy(d_matrix, flat_matrix.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
    
    inverse_2x2_kernel<<<1, 1>>>(d_matrix, d_inverse, d_invertible);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_invertible, d_invertible, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_invertible == 0) {
        cudaFree(d_matrix);
        cudaFree(d_inverse);
        cudaFree(d_invertible);
        return {};
    }
    
    cudaMemcpy(h_inverse.data(), d_inverse, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_matrix);
    cudaFree(d_inverse);
    cudaFree(d_invertible);
    
    return {
        {h_inverse[0], h_inverse[1]},
        {h_inverse[2], h_inverse[3]}
    };
}
