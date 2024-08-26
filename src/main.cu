#include "general.h"
#include "Mandelbrot.h"
#include "Burning_ship.h"
#include "Sierpinski.h"
#include "Julia.h"
#include "Koch.h"

int main() {
    sf::RenderWindow window(sf::VideoMode(width_window, height_window), "Fractals");
    sf::Image image;
    image.create(width_window, height_window);

    float offset_X = 0.0f;
    float offset_Y = 0.0f;
    float scale = 4.0f / width_window;

    // Выбор фрактала: закомментируйте одну из следующих строк для переключения между фракталами
    draw_fractal(image, offset_X, offset_Y, scale, mandelbrot_kernel);
    //draw_fractal(image, offset_X, offset_Y, scale, burning_ship_kernel);
    //draw_fractal_with_depth(image, 5, sierpinski_kernel);
    //draw_fractal(image, offset_X, offset_Y, scale, julia_kernel);
    //draw_koch_fractal(image, offset_X, offset_Y, scale);

    sf::Texture texture;
    texture.loadFromImage(image);

    sf::Sprite sprite;
    sprite.setTexture(texture);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            } else if (event.type == sf::Event::MouseWheelScrolled) {
                if (event.mouseWheelScroll.delta > 0) {
                    scale /= 1.1f;
                } else {
                    scale *= 1.1f;
                }
                draw_fractal(image, offset_X, offset_Y, scale, mandelbrot_kernel);
                //draw_fractal(image, offset_X, offset_Y, scale, burning_ship_kernel);
                //draw_fractal_with_depth(image, 5, sierpinski_kernel);
                //draw_fractal(image, offset_X, offset_Y, scale, julia_kernel);
                //draw_koch_fractal(image, offset_X, offset_Y, scale);
                texture.loadFromImage(image);
                sprite.setTexture(texture);
            } else if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    auto mouse_X = event.mouseButton.x;
                    auto mouse_Y = event.mouseButton.y;
                    offset_X += (mouse_X - width_window / 2.0f) * scale;
                    offset_Y += (mouse_Y - height_window / 2.0f) * scale;
                    draw_fractal(image, offset_X, offset_Y, scale, mandelbrot_kernel);
                    //draw_fractal(image, offset_X, offset_Y, scale, burning_ship_kernel);
                    //draw_fractal_with_depth(image, 5, sierpinski_kernel);
                    //draw_fractal(image, offset_X, offset_Y, scale, julia_kernel);
                    //draw_koch_fractal(image, offset_X, offset_Y, scale);
                    texture.loadFromImage(image);
                    sprite.setTexture(texture);
                }
            }
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }

    return 0;
}
