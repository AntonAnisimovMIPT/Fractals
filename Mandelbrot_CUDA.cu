#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <cuComplex.h>

const int width_window = 800;
const int height_window = 800;
const int number_iterations = 100;

__device__ int mandelbrot(cuFloatComplex c) {
    cuFloatComplex z = make_cuFloatComplex(0.0f, 0.0f);
    int n = 0;
    while (cuCabsf(z) <= 2.0f && n < number_iterations) {
        z = cuCaddf(cuCmulf(z, z), c);
        ++n;
    }
    return n;
}

__global__ void mandelbrot_kernel(sf::Uint8* pixels, float offset_X, float offset_Y, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width_window && y < height_window) {
        float real = (x - width_window / 2.0f) * scale + offset_X;
        float imag = (y - height_window / 2.0f) * scale + offset_Y;
        cuFloatComplex c = make_cuFloatComplex(real, imag);
        int depth = mandelbrot(c);

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
        pixels[index + 3] = 255;  // alpha channel
    }
}

void draw_mandelbrot(sf::Image &image, float offset_X, float offset_Y, float scale) {
    sf::Uint8* pixels = new sf::Uint8[width_window * height_window * 4];

    sf::Uint8* d_pixels;
    cudaMalloc(&d_pixels, width_window * height_window * 4 * sizeof(sf::Uint8));

    dim3 blockSize(16, 16);
    dim3 gridSize((width_window + blockSize.x - 1) / blockSize.x, (height_window + blockSize.y - 1) / blockSize.y);
    mandelbrot_kernel<<<gridSize, blockSize>>>(d_pixels, offset_X, offset_Y, scale);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, width_window * height_window * 4 * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);

    image.create(width_window, height_window, pixels);
    delete[] pixels;
}

int main() {
    sf::RenderWindow window(sf::VideoMode(width_window, height_window), "Mandelbrot Set");
    sf::Image image;
    image.create(width_window, height_window);

    float offset_X = 0.0f;
    float offset_Y = 0.0f;
    float scale = 4.0f / width_window;

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
                    scale /= 1.1f;
                } else {
                    scale *= 1.1f;
                }
                draw_mandelbrot(image, offset_X, offset_Y, scale);
                texture.loadFromImage(image);
                sprite.setTexture(texture);
            } else if (event.type == sf::Event::MouseButtonPressed) {
                if (event.mouseButton.button == sf::Mouse::Left) {
                    auto mouse_X = event.mouseButton.x;
                    auto mouse_Y = event.mouseButton.y;
                    offset_X += (mouse_X - width_window / 2.0f) * scale;
                    offset_Y += (mouse_Y - height_window / 2.0f) * scale;
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