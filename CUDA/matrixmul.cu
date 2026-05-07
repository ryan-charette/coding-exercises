#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

std::vector<std::vector<float>> matrixmul(
    const std::vector<std::vector<float>>& a,
    const std::vector<std::vector<float>>& b
) {
    int M = a.size();
    int K = a[0].size();
    int K2 = b.size();
    int N = b[0].size();
    
    // Check dimension compatibility
    if (K != K2) {
        return {{-1}};
    }
    
    // Flatten matrices
    std::vector<float> flat_a(M * K);
    std::vector<float> flat_b(K * N);
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            flat_a[i * K + j] = a[i][j];
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            flat_b[i * N + j] = b[i][j];
        }
    }
    
    float *d_a, *d_b, *d_c;
    std::vector<float> flat_c(M * N);
    
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));
    
    cudaMemcpy(d_a, flat_a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, flat_b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(flat_c.data(), d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Reshape to 2D
    std::vector<std::vector<float>> result(M, std::vector<float>(N));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = flat_c[i * N + j];
        }
    }
    
    return result;
}
