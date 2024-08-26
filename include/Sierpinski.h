#ifndef SIERPINSKI_H
#define SIERPINSKI_H

#include <SFML/Graphics.hpp>

__global__ void sierpinski_kernel(sf::Uint8* pixels, int depth);

void draw_sierpinski(sf::Image &image, int depth);

#endif
