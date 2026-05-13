#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA Kernel
__global__ void add(int* A, int* B, int* C, int size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

// Initialize vector
void initialize(int* vector, int size) {

    for (int i = 0; i < size; i++) {
        vector[i] = rand() % 10;
    }
}

// Print vector
void print(int* vector, int size) {

    for (int i = 0; i < size; i++) {
        cout << vector[i] << " ";
    }

    cout << endl;
}

int main() {

    int N = 4;

    int *A, *B, *C;

    size_t vectorBytes = N * sizeof(int);

    // Host memory allocation
    A = new int[N];
    B = new int[N];
    C = new int[N];

    // Initialize vectors
    initialize(A, N);
    initialize(B, N);

    cout << "Vector A: ";
    print(A, N);

    cout << "Vector B: ";
    print(B, N);

    // Device pointers
    int *X, *Y, *Z;

    // Allocate GPU memory
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);

    // Copy data from CPU to GPU
    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);

    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    int threadsPerBlock = 256;

    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);

    cudaDeviceSynchronize();

    // Copy result back to CPU
    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);

    cout << "Vector Addition Result: ";

    print(C, N);

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    return 0;
}