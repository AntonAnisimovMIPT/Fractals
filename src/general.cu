#include "general.h"
#include <cuda_runtime.h>

void draw_fractal(sf::Image &image, float offset_X, float offset_Y, float scale, 
                  void (*fractal_kernel)(sf::Uint8*, float, float, float)) {
    sf::Uint8* pixels = new sf::Uint8[width_window * height_window * 4];

    sf::Uint8* d_pixels;
    cudaMalloc(&d_pixels, width_window * height_window * 4 * sizeof(sf::Uint8));

    dim3 blockSize(16, 16);
    dim3 gridSize((width_window + blockSize.x - 1) / blockSize.x, (height_window + blockSize.y - 1) / blockSize.y);
    fractal_kernel<<<gridSize, blockSize>>>(d_pixels, offset_X, offset_Y, scale);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, width_window * height_window * 4 * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);

    image.create(width_window, height_window, pixels);
    delete[] pixels;
}
