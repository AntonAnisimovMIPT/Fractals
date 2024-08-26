#include "Koch.h"
#include "general.h"

__device__ void koch_line(sf::Uint8* pixels, float x1, float y1, float x2, float y2, int depth) {
    if (depth == 0) {
        int startX = min(x1, x2);
        int endX = max(x1, x2);
        for (int x = startX; x <= endX; ++x) {
            float t = (x - x1) / (x2 - x1);
            int y = y1 * (1 - t) + y2 * t;
            int index = (x + y * width_window) * 4;
            pixels[index] = 255;
            pixels[index + 1] = 255;
            pixels[index + 2] = 255;
            pixels[index + 3] = 255; 
        }
    } else {
        float dx = (x2 - x1) / 3.0f;
        float dy = (y2 - y1) / 3.0f;
        float x3 = x1 + dx;
        float y3 = y1 + dy;
        float x5 = x2 - dx;
        float y5 = y2 - dy;

        float x4 = 0.5f * (x1 + x2) + sqrt(3.0f) * (y1 - y2) / 6.0f;
        float y4 = 0.5f * (y1 + y2) + sqrt(3.0f) * (x2 - x1) / 6.0f;

        koch_line(pixels, x1, y1, x3, y3, depth - 1);
        koch_line(pixels, x3, y3, x4, y4, depth - 1);
        koch_line(pixels, x4, y4, x5, y5, depth - 1);
        koch_line(pixels, x5, y5, x2, y2, depth - 1);
    }
}

__global__ void koch_kernel(sf::Uint8* pixels, int depth, float x1, float y1, float x2, float y2) {
    koch_line(pixels, x1, y1, x2, y2, depth);
}

void draw_koch(sf::Image &image, int depth) {
    sf::Uint8* pixels = new sf::Uint8[width_window * height_window * 4]();

    sf::Uint8* d_pixels;
    cudaMalloc(&d_pixels, width_window * height_window * 4 * sizeof(sf::Uint8));
    cudaMemcpy(d_pixels, pixels, width_window * height_window * 4 * sizeof(sf::Uint8), cudaMemcpyHostToDevice);

    koch_kernel<<<1, 1>>>(d_pixels, depth, 100.0f, 400.0f, 700.0f, 400.0f);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, width_window * height_window * 4 * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);

    image.create(width_window, height_window, pixels);
    delete[] pixels;
}

void draw_koch_fractal(sf::Image &image, float offsetX, float offsetY, float scale) {
    int depth = 10; // Уровень рекурсии, можно настраивать
    draw_koch(image, depth);
}
