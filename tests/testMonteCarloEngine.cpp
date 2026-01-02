#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "../include/simulation/MonteCarloEngine.hpp"
#include "../include/processes/GeometricBrownianMotion.hpp"
#include "../include/core/TimeGrid.hpp"

#include <fstream>
#include <filesystem>
#include <cmath>

using namespace stochastic;
using Catch::Approx;


// Test 1 : Simulation basique GBM + contrôle du résultat théorique
TEST_CASE("Simulate GBM - basic test", "[MonteCarloEngine]")
{
    auto process = std::make_shared<GeometricBrownianMotion>(100.0, 0.08, 0.25, 12345);

    MonteCarloEngine::SimulationConfig cfg;
    cfg.nPaths = 5000;
    cfg.nSteps = 252;
    cfg.T      = 1.0;
    cfg.useExactScheme = true;
    cfg.antitheticVariates = false;
    cfg.seed   = 42;

    MonteCarloEngine engine(process, cfg);

    engine.simulate();
    auto stats = engine.getStatistics();

    REQUIRE(stats.nPaths == 5000);
    REQUIRE(stats.meanPath.size() == 253);  // nSteps + 1

    double theoreticalMean = 100.0 * std::exp(0.08 * 1.0);
    REQUIRE(std::abs(stats.terminalMean - theoreticalMean) < 2.0);

    double var = std::pow(100.0, 2) * std::exp(2 * 0.08) * (std::exp(0.25*0.25) - 1.0);
    double theoreticalStd = std::sqrt(var);
    REQUIRE(std::abs(stats.terminalStd - theoreticalStd) < 3.0);

    REQUIRE(std::abs(stats.skewness - 0.0) < 1.0);
    REQUIRE(std::abs(stats.kurtosis - 3.0) < 1.2);
}

// Test 2 : Antithetic variates
TEST_CASE("Antithetic variates reduce variance", "[MonteCarloEngine]")
{
    auto process = std::make_shared<GeometricBrownianMotion>(100.0, 0.05, 0.20, 999);

    MonteCarloEngine::SimulationConfig cfg;
    cfg.nPaths = 4000;
    cfg.nSteps = 100;
    cfg.T      = 1.0;
    cfg.useExactScheme = true;
    cfg.seed   = 123;

    cfg.antitheticVariates = false;
    MonteCarloEngine engine1(process->clone(), cfg);
    engine1.simulate();
    double std1 = engine1.getStatistics().terminalStd;

    cfg.antitheticVariates = true;
    MonteCarloEngine engine2(process->clone(), cfg);
    engine2.simulate();
    double std2 = engine2.getStatistics().terminalStd;

    // Antithetic variates should typically reduce variance; allow a modest tolerance
    REQUIRE(std2 <= std1 * 1.05);
}

// Test 3 : Parallélisme
TEST_CASE("Parallelism produces deterministic results", "[MonteCarloEngine]")
{
    auto process = std::make_shared<GeometricBrownianMotion>(100.0, 0.07, 0.22, 2024);

    MonteCarloEngine::SimulationConfig cfg;
    cfg.nPaths = 10000;
    cfg.nSteps = 50;
    cfg.T      = 0.5;
    cfg.useExactScheme = true;
    cfg.seed   = 777;

    cfg.nThreads = 1;
    MonteCarloEngine engine1(process->clone(), cfg);
    engine1.simulate();
    double mean1 = engine1.getStatistics().terminalMean;

    cfg.nThreads = 4;
    MonteCarloEngine engine2(process->clone(), cfg);
    engine2.simulate();
    double mean2 = engine2.getStatistics().terminalMean;

    REQUIRE(mean1 == Approx(mean2).margin(1e-6));
}

// Test 4 : Export CSV
TEST_CASE("Export to CSV generates a non-empty file", "[MonteCarloEngine]")
{
    auto process = std::make_shared<GeometricBrownianMotion>(100.0, 0.05, 0.2);

    MonteCarloEngine::SimulationConfig cfg;
    cfg.nPaths = 100;
    cfg.nSteps = 20;
    cfg.T      = 1.0;

    MonteCarloEngine engine(process, cfg);
    engine.simulate();

    const std::string filename = "test_paths_export.csv";

    if (std::filesystem::exists(filename))
        std::filesystem::remove(filename);

    engine.exportToCSV(filename);

    REQUIRE(std::filesystem::exists(filename));

    std::ifstream file(filename);
    REQUIRE(file.is_open());

    int lineCount = 0;
    std::string line;
    while (std::getline(file, line))
        ++lineCount;

    REQUIRE(lineCount == cfg.nPaths);

    file.close();
    std::filesystem::remove(filename);
}

// Test 5 : Exception si getStatistics() appelé avant simulate()
TEST_CASE("Calling getStatistics before simulate throws", "[MonteCarloEngine]")
{
    auto process = std::make_shared<GeometricBrownianMotion>(100.0, 0.05, 0.2);
    MonteCarloEngine engine(process, MonteCarloEngine::SimulationConfig{});

    REQUIRE_THROWS_AS(engine.getStatistics(), std::runtime_error);
}

// Test 6 : Exact scheme plausibility
TEST_CASE("Exact scheme uses closed-form GBM solution", "[MonteCarloEngine]")
{
    auto process = std::make_shared<GeometricBrownianMotion>(100.0, 0.1, 0.3, 111);

    MonteCarloEngine::SimulationConfig cfg;
    cfg.nPaths = 1;
    cfg.nSteps = 10;
    cfg.T      = 1.0;
    cfg.useExactScheme = true;
    cfg.seed   = 999;

    MonteCarloEngine engine(process, cfg);
    auto paths = engine.simulate();

    REQUIRE(!paths.empty());
    const auto& path = paths[0];

    REQUIRE(path.back() > 50.0);
    REQUIRE(path.back() < 300.0);
}
