#include "Julia.h"
#include "general.h"
#include <cuComplex.h>

__device__ int julia(cuFloatComplex z, cuFloatComplex c) {
    int n = 0;
    while (cuCabsf(z) <= 2.0f && n < number_iterations) {
        z = cuCaddf(cuCmulf(z, z), c);
        ++n;
    }
    return n;
}

__global__ void julia_kernel(sf::Uint8* pixels, float cRe, float cIm, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width_window && y < height_window) {
        float real = (x - width_window / 2.0f) * scale;
        float imag = (y - height_window / 2.0f) * scale;
        cuFloatComplex z = make_cuFloatComplex(real, imag);
        cuFloatComplex c = make_cuFloatComplex(cRe, cIm);
        int depth = julia(z, c);

        int index = (x + y * width_window) * 4;
        if (depth == number_iterations) {
            pixels[index] = 0;
            pixels[index + 1] = 0;
            pixels[index + 2] = 0;
        } else {
            pixels[index] = static_cast<sf::Uint8>(255 * (depth % 32) / 31);
            pixels[index + 1] = static_cast<sf::Uint8>(255 * (depth % 64) / 63);
            pixels[index + 2] = static_cast<sf::Uint8>(255 * (depth % 128) / 127);
        }
        pixels[index + 3] = 255;  // alpha channel
    }
}

void draw_julia(sf::Image &image, float cRe, float cIm, float scale) {
    sf::Uint8* pixels = new sf::Uint8[width_window * height_window * 4];

    sf::Uint8* d_pixels;
    cudaMalloc(&d_pixels, width_window * height_window * 4 * sizeof(sf::Uint8));

    dim3 blockSize(16, 16);
    dim3 gridSize((width_window + blockSize.x - 1) / blockSize.x, (height_window + blockSize.y - 1) / blockSize.y);
    julia_kernel<<<gridSize, blockSize>>>(d_pixels, cRe, cIm, scale);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, width_window * height_window * 4 * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);

    image.create(width_window, height_window, pixels);
    delete[] pixels;
}
