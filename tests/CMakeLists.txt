cmake_minimum_required(VERSION 3.28)
project(stochastic_tests)

# activer tests CTest
enable_testing()

# --- Catch2 ---
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.4.0  # tu peux mettre la dernière version stable
)
FetchContent_MakeAvailable(Catch2)

# --- fichiers sources des tests ---
file(GLOB TEST_SOURCES *.cpp)

# --- exécutable des tests ---
add_executable(unit_tests ${TEST_SOURCES})

# lier la librairie principale + Catch2
target_link_libraries(unit_tests PRIVATE
    stochastic_framework
    Catch2::Catch2WithMain
)

# --- découverte automatique des tests Catch2 ---
include(CTest)
include(Catch)
catch_discover_tests(unit_tests)
