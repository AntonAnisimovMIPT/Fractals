#include <SFML/Graphics.hpp>
#include <complex>

const int width_window = 800;
const int height_window = 800;
const int number_iterations = 100; // пощадите компьютер

auto get_color(int depth) {

    if (depth == number_iterations) {
        return sf::Color::Black;
    }

    // разные множители и знаменатели сделаный для градиентного эффекта
    auto r = static_cast<sf::Uint8>(255 * (depth % 32) / 31);
    auto g = static_cast<sf::Uint8>(255 * (depth % 64) / 63);
    auto b = static_cast<sf::Uint8>(255 * (depth % 128) / 127);

    return sf::Color(r, g, b);
}

auto mandelbrot(const std::complex<float> &c) {

    std::complex<float> z = 0;
    auto n = 0;
    while (abs(z) <= 2 && n < number_iterations) {
        z = z * z + c;
        ++n;
    }

    return n;
}

auto draw_mandelbrot(sf::Image &image, float offset_X, float offset_Y, float scale) {

    for (int x = 0; x < width_window; ++x) {
        for (int y = 0; y < height_window; ++y) {
            auto real = (x - width_window / 2.0) * scale + offset_X;
            auto imag = (y - height_window / 2.0) * scale + offset_Y;
            std::complex<float> c(real, imag);
            auto depth = mandelbrot(c);
            image.setPixel(x, y, get_color(depth));
        }
    }
}

int main() {
    sf::RenderWindow window(sf::VideoMode(width_window, height_window), "Mandelbrot Set");
    sf::Image image;
    image.create(width_window, height_window);

    auto offset_X = 0.0f;
    auto offset_Y = 0.0f;
    auto scale = 4.0f / width_window;

    draw_mandelbrot(image, offset_X, offset_Y, scale);

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
                    scale /= 1.1;
                } else {
                    scale *= 1.1;
                }

                draw_mandelbrot(image, offset_X, offset_Y, scale);
                texture.loadFromImage(image);
                sprite.setTexture(texture);

            } else if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {

                    auto mouse_X = event.mouseButton.x;
                    auto mouse_Y = event.mouseButton.y;
                    offset_X += (mouse_X - width_window / 2.0) * scale;
                    offset_Y += (mouse_Y - height_window / 2.0) * scale;
                    draw_mandelbrot(image, offset_X, offset_Y, scale);
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
