#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>

#include "../include/calibration/MCMCCalibrator.hpp"
#include "../include/processes/GeometricBrownianMotion.hpp"
#include "../include/core/TimeGrid.hpp"

using namespace stochastic;
using Catch::Approx;

// ============================================================================
// Test 1 : Calibration complète sur un GBM avec données simulées
// ============================================================================
TEST_CASE("MCMCCalibrator - Calibrate GBM with simulated data", "[MCMCCalibrator]")
{
    GeometricBrownianMotion trueProcess(100.0, 0.08, 0.25, 12345);
    TimeGrid grid(0.0, 2.0, 504);

    auto pricePath = trueProcess.simulatePathExact(grid);

    std::vector<double> logReturns;
    logReturns.reserve(pricePath.size() - 1);

    for (size_t i = 1; i < pricePath.size(); ++i)
        logReturns.push_back(std::log(pricePath[i] / pricePath[i - 1]));

    TimeSeriesData tsData;
    tsData.path = logReturns;
    tsData.dt   = 1.0 / 252.0;

    auto processToCalibrate = std::make_shared<GeometricBrownianMotion>(
        100.0,
        0.03,
        0.15,
        42
    );

    MCMCCalibrator::MCMCConfig config;
    config.nIterations = 8000;
    config.burnIn      = 2000;
    config.thinning    = 5;
    config.adaptiveProposal = true;

    config.priors["mu"]    = {MCMCCalibrator::Prior::Type::NORMAL,    {0.0, 0.3}};
    config.priors["sigma"] = {MCMCCalibrator::Prior::Type::LOGNORMAL, {-2.0, 0.6}};

    MCMCCalibrator calibrator(processToCalibrate, tsData, config);
    auto result = calibrator.calibrateFromMarketData();

    REQUIRE(result.acceptanceRate > 0.05);
    REQUIRE(result.acceptanceRate < 0.70);

    const double trueMu    = 0.08;
    const double trueSigma = 0.25;

    double estimatedMu    = result.meanParams.at("mu");
    double estimatedSigma = result.meanParams.at("sigma");

    REQUIRE(estimatedMu    == Approx(trueMu).margin(0.04));
    REQUIRE(estimatedSigma == Approx(trueSigma).margin(0.06));

    auto [mu_low, mu_high]         = result.credibleIntervals95.at("mu");
    auto [sigma_low, sigma_high]   = result.credibleIntervals95.at("sigma");

    REQUIRE(mu_low    < trueMu);
    REQUIRE(mu_high   > trueMu);
    REQUIRE(sigma_low < trueSigma);
    REQUIRE(sigma_high > trueSigma);

    REQUIRE(std::isfinite(result.logLikelihood));
    REQUIRE(result.logLikelihood > -1e6);
}

// ============================================================================
// Test 2 : LogLikelihood cohérent
// ============================================================================
TEST_CASE("MCMCCalibrator - LogLikelihood consistency", "[MCMCCalibrator]")
{
    auto process = std::make_shared<GeometricBrownianMotion>(100.0, 0.07, 0.22);

    std::vector<double> obs = {0.001, -0.002, 0.003, 0.0005, -0.0015};
    double dt = 1.0 / 252.0;

    double llTrue  = process->logLikelihood(obs, dt);

    process->setMu(0.15);
    process->setSigma(0.35);
    double llWrong = process->logLikelihood(obs, dt);

    REQUIRE(llTrue > llWrong);
}

// ============================================================================
// Test 3 : Priors très informatifs → domination du prior
// ============================================================================
TEST_CASE("MCMCCalibrator - Strong prior dominates likelihood", "[MCMCCalibrator]")
{
    auto process = std::make_shared<GeometricBrownianMotion>(100.0, 0.05, 0.2);

    TimeSeriesData data;
    data.path = {0.01, 0.02, -0.01};
    data.dt   = 1.0 / 252.0;

    MCMCCalibrator::MCMCConfig config;
    config.nIterations = 5000;
    config.burnIn      = 1000;
    config.thinning    = 5;

    config.priors["mu"]    = {MCMCCalibrator::Prior::Type::NORMAL, {0.03, 0.01}};
    config.priors["sigma"] = {MCMCCalibrator::Prior::Type::LOGNORMAL, {std::log(0.18), 0.1}};

    MCMCCalibrator calibrator(process, data, config);
    auto result = calibrator.calibrateFromMarketData();

    REQUIRE(result.meanParams.at("mu")    == Approx(0.03).margin(0.02));
    REQUIRE(result.meanParams.at("sigma") == Approx(0.18).margin(0.03));
}

// ============================================================================
// Test 4 : sigma ≤ 0 → logLik = -inf
// ============================================================================
TEST_CASE("MCMCCalibrator - Invalid parameters give -inf likelihood", "[MCMCCalibrator]")
{
    auto process = std::make_shared<GeometricBrownianMotion>(100.0, 0.05, 0.2);

    std::vector<double> obs = {0.001, 0.002};
    double dt = 1.0 / 252.0;

    REQUIRE_THROWS_AS(process->setSigma(0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(process->setSigma(-0.1), std::invalid_argument);
}
