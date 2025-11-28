cmake_minimum_required(VERSION 3.10)

project(MonProjet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ajouter le dossier include/
include_directories(include)

# Récupérer les sources C++ (récursivement)
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# Ajouter l'exécutable
add_executable(monprogramme
    main.cpp
    ${SRC_FILES}
)

# Chercher libcurl et Lier libcurl
find_package(CURL REQUIRED)
target_link_libraries(monprogramme PRIVATE CURL::libcurl)

# Chercher nlohmann_json et le lier
find_package(nlohmann_json REQUIRED)
target_link_libraries(monprogramme PRIVATE nlohmann_json::nlohmann_json)

#  Chercher Eigen et le lier
find_package(Eigen3 REQUIRED)
target_link_libraries(monprogramme PRIVATE Eigen3::Eigen)