#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void compute_means_kernel(
    const float* data,
    float* means,
    int n_features,
    int n_observations
) {
    int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature < n_features) {
        float sum = 0.0f;
        for (int i = 0; i < n_observations; i++) {
            sum += data[feature * n_observations + i];
        }
        means[feature] = sum / n_observations;
    }
}

__global__ void compute_covariance_kernel(
    const float* data,
    const float* means,
    float* covariance,
    int n_features,
    int n_observations
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_features && j < n_features) {
        float cov = 0.0f;
        for (int k = 0; k < n_observations; k++) {
            float diff_i = data[i * n_observations + k] - means[i];
            float diff_j = data[j * n_observations + k] - means[j];
            cov += diff_i * diff_j;
        }
        covariance[i * n_features + j] = cov / (n_observations - 1);
    }
}

std::vector<std::vector<float>> calculate_covariance_matrix(const std::vector<std::vector<float>>& vectors) {
    int n_features = vectors.size();
    int n_observations = vectors[0].size();
    
    // Flatten data (row-major: feature x observation)
    std::vector<float> flat_data(n_features * n_observations);
    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < n_observations; j++) {
            flat_data[i * n_observations + j] = vectors[i][j];
        }
    }
    
    float *d_data, *d_means, *d_covariance;
    
    cudaMalloc(&d_data, n_features * n_observations * sizeof(float));
    cudaMalloc(&d_means, n_features * sizeof(float));
    cudaMalloc(&d_covariance, n_features * n_features * sizeof(float));
    
    cudaMemcpy(d_data, flat_data.data(), n_features * n_observations * sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute means
    int threads1 = 256;
    int blocks1 = (n_features + threads1 - 1) / threads1;
    compute_means_kernel<<<blocks1, threads1>>>(d_data, d_means, n_features, n_observations);
    cudaDeviceSynchronize();
    
    // Compute covariance
    dim3 block(16, 16);
    dim3 grid((n_features + 15) / 16, (n_features + 15) / 16);
    compute_covariance_kernel<<<grid, block>>>(d_data, d_means, d_covariance, n_features, n_observations);
    cudaDeviceSynchronize();
    
    std::vector<float> flat_cov(n_features * n_features);
    cudaMemcpy(flat_cov.data(), d_covariance, n_features * n_features * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_means);
    cudaFree(d_covariance);
    
    // Reshape to 2D
    std::vector<std::vector<float>> result(n_features, std::vector<float>(n_features));
    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < n_features; j++) {
            result[i][j] = flat_cov[i * n_features + j];
        }
    }
    
    return result;
}
