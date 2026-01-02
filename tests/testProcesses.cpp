#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "../include/processes/GeometricBrownianMotion.hpp"
#include "../include/processes/HestonModel.hpp"
#include "../include/processes/CoxIngersollRoss.hpp"
#include "../include/processes/OrnsteinUhlenbeck.hpp"
#include "../include/processes/FractionalBrownianMotion.hpp"
#include "../include/processes/LevyJumpDiffusion.hpp"
#include "../include/core/TimeGrid.hpp"

using Catch::Approx;
using namespace stochastic;


// GMB
TEST_CASE("GBM next step is positive", "[GBM]") {
    stochastic::GeometricBrownianMotion gbm(100.0, 0.05, 0.2, 42);
    double dt = 1.0 / 252.0;
    double next = gbm.nextStep(dt);

    REQUIRE(next > 0.0);
}

TEST_CASE("GBM simulatePathExact works", "[GBM]") {
    stochastic::GeometricBrownianMotion gbm(100.0, 0.05, 0.2, 42);
    TimeGrid grid(0.0, 1.0, 5);

    auto path = gbm.simulatePathExact(grid);

    REQUIRE(path.size() == 6);  // 5 steps + initial point
    REQUIRE(gbm.theoreticalMean(1.0) == Approx(100 * std::exp(0.05)).margin(1e-2));
}

// Heston
TEST_CASE("Heston parameters are validated/corrected", "[Heston]") {
    stochastic::HestonModel heston(
        100.0, 0.05, -1.0, 0.04, 0.3, -0.7, 0.04, 42
    );

    heston.validateParameters();
    REQUIRE(heston.getKappa() > 0.0);
}

TEST_CASE("Heston next step keeps variance positive", "[Heston]") {
    stochastic::HestonModel h(100, 0.05, 2.0, 0.04, 0.3, -0.5, 0.04, 42);

    double spot = h.nextStep(1.0/252.0);
    double var = h.getCurrentVariance();

    REQUIRE(var > 0.0);
    REQUIRE(spot > 0.0);
}

// CIR
TEST_CASE("CIR next step stays positive", "[CIR]") {
    stochastic::CoxIngersollRoss cir(
        0.04, 1.0, 0.05, 0.1, 42
    );

    double r1 = cir.nextStep(1.0 / 252.0);
    REQUIRE(r1 > 0.0);
}

TEST_CASE("CIR simulatePath returns correct size", "[CIR]") {
    stochastic::CoxIngersollRoss cir(0.03, 1.0, 0.04, 0.2, 42);
    TimeGrid grid(0.0, 1.0, 10);
    auto path = cir.simulatePath(grid);

    REQUIRE(path.size() == 11);
}

// OU
TEST_CASE("OU next step is finite", "[OU]") {
    stochastic::OrnsteinUhlenbeck ou(0.0, 1.0, 1.0, 0.3, 42);

    double x1 = ou.nextStep(0.1);
    REQUIRE(std::isfinite(x1));
}

TEST_CASE("OU simulatePath returns correct size", "[OU]") {
    stochastic::OrnsteinUhlenbeck ou(0.0, 2.0, 1.0, 0.2, 42);
    TimeGrid grid(0.0, 2.0, 20);
    auto path = ou.simulatePath(grid);

    REQUIRE(path.size() == 21);
}

// fBM
TEST_CASE("fBM Hurst parameter is in (0,1)", "[fBM]") {
    stochastic::FractionalBrownianMotion fbm(0.75, 42);

    REQUIRE(fbm.getH() > 0.0);
    REQUIRE(fbm.getH() < 1.0);
}

TEST_CASE("fBM simulatePath returns correct size", "[fBM]") {
    stochastic::FractionalBrownianMotion fbm(0.7, 42);
    TimeGrid grid(0.0, 1.0, 8);
    auto path = fbm.simulatePath(grid);

    REQUIRE(path.size() == 9);
}

// Levy
TEST_CASE("Levy next step stays positive", "[Levy]") {
    stochastic::LevyJumpDiffusion lev(
        100.0, 0.05, 0.2, 0.1, -0.1, 0.2, 42
    );

    double next = lev.nextStep(1.0/252.0);
    REQUIRE(next > 0.0);
}

TEST_CASE("Levy simulatePath returns correct size", "[Levy]") {
    stochastic::LevyJumpDiffusion lev(100, 0.05, 0.2, 0.2, -0.05, 0.2, 42);
    TimeGrid grid(0.0, 1.0, 10);
    auto path = lev.simulatePath(grid);

    REQUIRE(path.size() == 11);
}
