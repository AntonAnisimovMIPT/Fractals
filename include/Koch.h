#ifndef KOCH_H
#define KOCH_H

#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include "general.h"

__device__ void koch_line(sf::Uint8* pixels, float x1, float y1, float x2, float y2, int depth);

__global__ void koch_kernel(sf::Uint8* pixels, int depth, float x1, float y1, float x2, float y2);

void draw_koch(sf::Image &image, int depth);

void draw_koch_fractal(sf::Image &image, float offsetX, float offsetY, float scale);

#endif 
