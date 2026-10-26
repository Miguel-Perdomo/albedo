cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core cpp/bindings.cpp)
target_include_directories(_core PRIVATE cpp)
if(MSVC)
  target_compile_options(_core PRIVATE /W4)
else()
  target_compile_options(_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
install(TARGETS _core DESTINATION albedo)
