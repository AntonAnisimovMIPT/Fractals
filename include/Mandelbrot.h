#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <SFML/Graphics.hpp>

__global__ void mandelbrot_kernel(sf::Uint8* pixels, float offset_X, float offset_Y, float scale);

#endif // MANDELBROT_H
