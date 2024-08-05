#include <stdio.h>

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        c[id] = a[id] + b[id];
}

void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int n = 100000;
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(int);

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    checkCudaError(cudaMalloc(&d_a, bytes), "Failed to allocate device memory for d_a");
    checkCudaError(cudaMalloc(&d_b, bytes), "Failed to allocate device memory for d_b");
    checkCudaError(cudaMalloc(&d_c, bytes), "Failed to allocate device memory for d_c");

    checkCudaError(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice), "Failed to copy data from host to device for d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice), "Failed to copy data from host to device for d_b");

    int blockSize = 1024;
    int gridSize = (int)ceil((float)n / blockSize);

    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    checkCudaError(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost), "Failed to copy data from device to host for d_c");

    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            printf("Error at index %d: %d + %d != %d\n", i, h_a[i], h_b[i], h_c[i]);
            return -1;
        }
    }

    printf("Test PASSED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
