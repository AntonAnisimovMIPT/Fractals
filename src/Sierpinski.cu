#include "Sierpinski.h"
#include "general.h"

__device__ void draw_triangle(sf::Uint8* pixels, int x, int y, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j <= i; ++j) {
            int index = (x + j + (y + i) * width_window) * 4;
            pixels[index] = 255;
            pixels[index + 1] = 255;
            pixels[index + 2] = 255;
            pixels[index + 3] = 255; // alpha channel
        }
    }
}

__global__ void sierpinski_kernel(sf::Uint8* pixels, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int size = width_window;

    for (int i = 0; i < depth; ++i) {
        int step = size / (1 << (i + 1));
        if ((x / step) % 2 == 1 && (y / step) % 2 == 1) {
            return;
        }
    }
    draw_triangle(pixels, x, y, 1);
}

void draw_sierpinski(sf::Image &image, int depth) {
    sf::Uint8* pixels = new sf::Uint8[width_window * height_window * 4];

    sf::Uint8* d_pixels;
    cudaMalloc(&d_pixels, width_window * height_window * 4 * sizeof(sf::Uint8));

    dim3 blockSize(16, 16);
    dim3 gridSize((width_window + blockSize.x - 1) / blockSize.x, (height_window + blockSize.y - 1) / blockSize.y);
    sierpinski_kernel<<<gridSize, blockSize>>>(d_pixels, depth);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, width_window * height_window * 4 * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);

    image.create(width_window, height_window, pixels);
    delete[] pixels;
}
