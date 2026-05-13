#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for matrix multiplication
__global__ void multiply(int* A, int* B, int* C, int size) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {

        int sum = 0;

        for (int i = 0; i < size; i++) {
            sum += A[row * size + i] * B[i * size + col];
        }

        C[row * size + col] = sum;
    }
}

// Initialize matrix
void initialize(int* matrix, int size) {

    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 10;
    }
}

// Print matrix
void print(int* matrix, int size) {

    for (int row = 0; row < size; row++) {

        for (int col = 0; col < size; col++) {
            cout << matrix[row * size + col] << " ";
        }

        cout << endl;
    }

    cout << endl;
}

int main() {

    int N = 2;

    int matrixSize = N * N;
    size_t matrixBytes = matrixSize * sizeof(int);

    // Host matrices
    int *A, *B, *C;

    A = new int[matrixSize];
    B = new int[matrixSize];
    C = new int[matrixSize];

    // Initialize matrices
    initialize(A, N);
    initialize(B, N);

    cout << "Matrix A:\n";
    print(A, N);

    cout << "Matrix B:\n";
    print(B, N);

    // Device matrices
    int *X, *Y, *Z;

    cudaMalloc(&X, matrixBytes);
    cudaMalloc(&Y, matrixBytes);
    cudaMalloc(&Z, matrixBytes);

    // Copy host to device
    cudaMemcpy(X, A, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, matrixBytes, cudaMemcpyHostToDevice);

    // Thread configuration
    int THREADS = 2;

    dim3 threads(THREADS, THREADS);

    dim3 blocks((N + THREADS - 1) / THREADS,
                (N + THREADS - 1) / THREADS);

    // Launch kernel
    multiply<<<blocks, threads>>>(X, Y, Z, N);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(C, Z, matrixBytes, cudaMemcpyDeviceToHost);

    cout << "Matrix Multiplication Result:\n";
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