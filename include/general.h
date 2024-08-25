#ifndef GENERAL_H
#define GENERAL_H

#include <SFML/Graphics.hpp>

__device__ const int width_window = 800;
__device__ const int height_window = 800;
__device__ const int number_iterations = 100;

void draw_fractal(sf::Image &image, float offset_X, float offset_Y, float scale, 
                  void (*fractal_kernel)(sf::Uint8*, float, float, float));

#endif // GENERAL_H
