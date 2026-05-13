#include <stdio.h>
#include <limits.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Kernel for Minimum Reduction
__global__ void reduceMin(int *input, int *output, int size) {

    __shared__ int sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < size)
        sdata[tid] = input[i];
    else
        sdata[tid] = INT_MAX;

    __syncthreads();

    // Reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {
            if (sdata[tid + stride] < sdata[tid]) {
                sdata[tid] = sdata[tid + stride];
            }
        }

        __syncthreads();
    }

    // Store result
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// Kernel for Maximum Reduction
__global__ void reduceMax(int *input, int *output, int size) {

    __shared__ int sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        sdata[tid] = input[i];
    else
        sdata[tid] = INT_MIN;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {
            if (sdata[tid + stride] > sdata[tid]) {
                sdata[tid] = sdata[tid + stride];
            }
        }

        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// Kernel for Sum Reduction
__global__ void reduceSum(int *input, int *output, int size) {

    __shared__ int sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        sdata[tid] = input[i];
    else
        sdata[tid] = 0;

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

int main() {

    const int SIZE = 256;

    int h_input[SIZE];

    // Initialize array
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = i + 1;
    }

    // Device pointers
    int *d_input, *d_min, *d_max, *d_sum;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_input, SIZE * sizeof(int));
    cudaMalloc((void**)&d_min, sizeof(int));
    cudaMalloc((void**)&d_max, sizeof(int));
    cudaMalloc((void**)&d_sum, sizeof(int));

    // Copy input to GPU
    cudaMemcpy(d_input, h_input, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernels
    reduceMin<<<1, BLOCK_SIZE>>>(d_input, d_min, SIZE);

    reduceMax<<<1, BLOCK_SIZE>>>(d_input, d_max, SIZE);

    reduceSum<<<1, BLOCK_SIZE>>>(d_input, d_sum, SIZE);

    // Host variables
    int min_result, max_result, sum_result;

    // Copy results back
    cudaMemcpy(&min_result, d_min, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(&max_result, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemcpy(&sum_result, d_sum, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate average
    float avg_result = (float)sum_result / SIZE;

    // Print results
    printf("Minimum Value : %d\n", min_result);

    printf("Maximum Value : %d\n", max_result);

    printf("Sum : %d\n", sum_result);

    printf("Average : %.2f\n", avg_result);

    // Free GPU memory
    cudaFree(d_input);

    cudaFree(d_min);

    cudaFree(d_max);

    cudaFree(d_sum);

    return 0;
}