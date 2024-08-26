#ifndef JULIA_H
#define JULIA_H

#include <SFML/Graphics.hpp>

__global__ void julia_kernel(sf::Uint8* pixels, float cRe, float cIm, float scale);
void draw_julia(sf::Image &image, float cRe, float cIm, float scale);

#endif 
