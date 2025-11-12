cmake_minimum_required(VERSION 3.15)
project(StochasticSimulation VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Include Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Include local headers
include_directories(include)

file(GLOB SOURCES "src/*.cpp")
add_executable(stochastic_sim ${SOURCES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(stochastic_sim PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen)
else()
    target_link_libraries(stochastic_sim PUBLIC Eigen3::Eigen)
endif()

