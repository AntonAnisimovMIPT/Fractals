cmake_minimum_required(VERSION 3.10)
project(Mandelbrot)

# Укажите стандарт C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Укажите путь к библиотекам SFML, если они установлены в нестандартное место
set(SFML_DIR "")

# Найдите библиотеки SFML
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Добавьте исполняемый файл
add_executable(Mandelbrot Mandelbrot.cpp)

# Подключите библиотеки SFML
target_link_libraries(Mandelbrot sfml-graphics sfml-window sfml-system)
