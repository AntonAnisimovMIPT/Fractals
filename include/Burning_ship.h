#ifndef BURNING_SHIP_H
#define BURNING_SHIP_H

#include <SFML/Graphics.hpp>

__global__ void burning_ship_kernel(sf::Uint8* pixels, float offset_X, float offset_Y, float scale);

#endif // BURNING_SHIP_H
