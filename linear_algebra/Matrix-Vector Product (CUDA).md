# Matrix-Vector Product (CUDA)

## Problem Statement
Given a matrix (a list of lists of floats) `matrix` with $m$ rows and $n$ columns, and a vector `vector`, compute `result = matrix @ vector`, where `result[i]` is the dot product of one row of `matrix` with `vector`:

```C++
result[i] = matrix[i][0] @ vector[0] + matrix[i][1] @ vector[1] + ... + matrix[i][n] @ vector[n]
```

If the matrix and vector do not have compatible dimensions, return an empty vector.

You may assume that the input matrix is a non-empty, non-jagged list of lists and that the vector is a non-empty list.

Example:
```C++
matrix = [
    [1, 2, 3],
    [4, 5, 6]
]

vector = [7, 8, 9]
```
Output:
```C++
[50, 122]
```
Explanation:
```C++
[
    1*7 + 2*8 + 3*9,   // 50
    4*7 + 5*8 + 6*9    // 122
]
```

## Code Skeleton
```C++
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrix_vector_product_kernel
(
    const float* matrix,
    const float* vector,
    float* result,
    int rows,
    int cols
) 
{

}

std::vector<float> matrix_vector_product
(
    const std::vector<std::vector<float>>& matrix, 
    const std::vector<float>& vector
) 
{
    return {};
}
```
## Intuition
On the CPU, we would usually do this with two loops:
```
for each row:
    dot = 0
    for each column:
        dot += matrix[row][col] * vector[col]
    result[row] = dot
```
The key observation is that each row's dot product is independent. 

That means:
```
result[0] does not depend on result[1]
result[1] does not depend on result[2]
...
```
So this is a good candidate for GPU parallelism. Instead of one CPU thread computing every row, we let many GPU threads work at the same time.

In this implementation, each CUDA thread computes one full row's dot product. So if the matrix has `rows` rows, we launch (at least) `rows` CUDA threads.

## CPU Baseline
The normal CPU solution would look like this:
```C++
std::vector<float> result(rows);

for (int r = 0; r < rows; r++)
{
    float dot = 0.0f;

    for (int c = 0; c < cols; c++)
    {
        dot += matrix[r][c] * vector[c];
    }

    result[r] = dot;
}
```
This solution has time complexity $O(mn)$, because every element of the matrix contributes to the dot product exactly once.

The space complexity is $O(1)$ because the only extra memory used is to store `dot` before it is added to the `result` array.

## GPU Kernel Function
The input matrix is of the form `std::vector<std::vector<float>>`, but CUDA device memory expects a contiguous block of memory. A `vector<vector<float>>` is not guaranteed to be stored as one continuous matrix. Each row may live in a different memory location. Therefore, we must flatten it into 
```C++
std::vector<float> flat_matrix(rows * cols);
```
Example:
```C++
[
    [1, 2, 3],
    [4, 5, 6]
]
```
becomes:
```C++
[1, 2, 3, 4, 5, 6]
```
Notice that when we do this,
```C++
matrix[r][c] = flat_matrix[r * cols + c]
```
### Thread Indexing
The global thread ID is given by
```C++
blockIdx.x * blockDim.x + threadIdx.x
```
The total number of launched threads is a multiple of `threadsPerBlock`, which may be more than the number of threads we need. We'll need to check that the thread corresponds to a row in `matrix`. 

With this in mind, we can go ahead and write our CUDA kernel function.

```C++
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrix_vector_product_kernel
(
    const float* matrix,
    const float* vector,
    float* result,
    int rows,
    int cols
) 
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows)
    {
        float dot = 0.0f;

        for (int c = 0; c < cols; c++)
        {
            dot += matrix[r * cols + c] * vector[c];
        }

        result[r] = dot;
    }
}
```
## Host Function
Now we need to implement the host-side wrapper. This function has several responsibilities:

### 1. Validate dimensions
In order to perform matrix-vector multiplication, every row of `matrix` must have the same number of elements as `vector`. Since our problem statement allows us to assume that `matrix` is non-empty and non-jagged, we can simply check:
```C++
if (matrix[0].size() != cols)
{
    return {};
}
```

### 2. Flatten Matrix
Now we need to convert the nested vector into contiguous memory for the kernel function.
```C++
std::vector<float> flat_matrix(rows * cols);
for (int r = 0; r < rows; r++) 
{
    for (int c = 0; c < cols; c++) 
    {
        flat_matrix[r * cols + c] = matrix[r][c];
    }
}
```

### 3. Kernel Management
There are several tasks that every CUDA program must do:

Allocate device memory:
```C++
float* d_matrix;
float* d_vector;
float* d_result;

cudaMalloc(&d_matrix, rows * cols * sizeof(float));
cudaMalloc(&d_vector, cols * sizeof(float));
cudaMalloc(&d_result, rows * sizeof(float));
```
Copy data to device:
```C++
cudaMemcpy
(
    d_matrix, 
    flat_matrix.data(),
    rows * cols * sizeof(float),
    cudaMemcpyHostToDevice
);

cudaMemcpy
(
    d_vector, 
    vector.data(),
    cols * sizeof(float),cudaMemcpyHostToDevice
);
```
Organize threads and blocks:
```C++
int threadsPerBlock = 256;
int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
```
Launch the kernel:
```C++
matrix_vector_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>
(
    d_matrix, 
    d_vector, 
    d_result, 
    rows, 
    cols
);
```
Copy the result back to the host:
```C++
cudaMemcpy
(
    result.data(), 
    d_result,
    rows * sizeof(float),
    cudaMemcpyDeviceToHost
);
```
Free device memory:
```C++
cudaFree(d_matrix);
cudaFree(d_vector);
cudaFree(d_result);
```
Return result:
```C++
return result;
```
Putting it all together, we get our host function:
```C++
std::vector<float> matrix_vector_product
(
    const std::vector<std::vector<float>>& matrix, 
    const std::vector<float>& vector
) 
{
    int rows = matrix.size();
    int cols = vector.size();
    std::vector<float> result(rows);

    if (matrix[0].size() != cols)
    {
        return {};
    }

    std::vector<float> flat_matrix(rows * cols);
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            flat_matrix[r * cols + c] = matrix[r][c];
        }
    }

    float* d_matrix;
    float* d_vector;
    float* d_result;

    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    cudaMalloc(&d_vector, cols * sizeof(float));
    cudaMalloc(&d_result, rows * sizeof(float));

    cudaMemcpy
    (
        d_matrix,
        flat_matrix.data(),
        rows * cols * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy
    (
        d_vector,
        vector.data(),
        cols * sizeof(float),cudaMemcpyHostToDevice
    );

    int threadsPerBlock = 256;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;

    matrix_vector_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>
    (
        d_matrix,
        d_vector,
        d_result,
        rows,
        cols
    );

    cudaMemcpy
    (
        result.data(),
        d_result,
        rows * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return result;
}
```
