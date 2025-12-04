#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath> 
#include "../include/core/TimeGrid.hpp"

using namespace stochastic;

TEST_CASE("TimeGrid construction", "[TimeGrid]") {
    TimeGrid grid(0.0, 1.0, 10);

    REQUIRE(grid.size() == 11);
    REQUIRE(std::abs(grid.getIncrement(0) - 0.1) < 1e-6);
    REQUIRE(std::abs(grid[10] - 1.0) < 1e-6);
}