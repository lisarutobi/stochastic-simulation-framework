cmake_minimum_required(VERSION 3.10)

project(MonProjet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------
# Include
# ----------------------------------------------------------
include_directories(include)

# ----------------------------------------------------------
# Sources du framework
# ----------------------------------------------------------
file(GLOB_RECURSE FRAMEWORK_SRC
    src/data/*.cpp
    src/simulation/*.cpp
    src/calibration/*.cpp
)

# ----------------------------------------------------------
# Bibliothèque du framework (MonteCarloEngine, loaders, etc.)
# ----------------------------------------------------------
add_library(stochastic_framework STATIC
    ${FRAMEWORK_SRC}
)

# ----------------------------------------------------------
# Dépendances externes
# ----------------------------------------------------------
find_package(CURL REQUIRED)
target_link_libraries(stochastic_framework PRIVATE CURL::libcurl)

find_package(nlohmann_json REQUIRED)
target_link_libraries(stochastic_framework PRIVATE nlohmann_json::nlohmann_json)

find_package(Eigen3 REQUIRED)
target_link_libraries(stochastic_framework PRIVATE Eigen3::Eigen)

# ----------------------------------------------------------
# Executable principal
# ----------------------------------------------------------
add_executable(monprogramme
    main.cpp
)

# Link du main contre la lib du framework
target_link_libraries(monprogramme PRIVATE stochastic_framework)
