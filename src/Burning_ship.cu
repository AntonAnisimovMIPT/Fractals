#include "Burning_ship.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "general.h"

__device__ int burning_ship(cuFloatComplex c) {
    cuFloatComplex z = make_cuFloatComplex(0.0f, 0.0f);
    int n = 0;
    while (cuCabsf(z) <= 2.0f && n < number_iterations) {
        z = make_cuFloatComplex(fabsf(cuCrealf(z)), fabsf(cuCimagf(z)));
        z = cuCaddf(cuCmulf(z, z), c);
        ++n;
    }
    return n;
}

__global__ void burning_ship_kernel(sf::Uint8* pixels, float offset_X, float offset_Y, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width_window && y < height_window) {
        float real = (x - width_window / 2.0f) * scale + offset_X;
        float imag = (y - height_window / 2.0f) * scale + offset_Y;
        cuFloatComplex c = make_cuFloatComplex(real, imag);
        int depth = burning_ship(c);

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
        pixels[index + 3] = 255;  
    }
}
