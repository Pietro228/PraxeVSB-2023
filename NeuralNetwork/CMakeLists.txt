cmake_minimum_required(VERSION 3.23)
project(NeauralNetwork_shape_recognition)

set(CMAKE_CXX_STANDARD 14)

add_executable(NeauralNetwork_shape_recognition main.cpp NeuralNetwork.cpp NeuralNetwork.h Layer.cpp Layer.h)
