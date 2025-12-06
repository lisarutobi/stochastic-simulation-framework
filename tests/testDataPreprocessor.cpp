#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>

#include "../include/data/DataPreprocessor.hpp"
#include "../include/data/MarketDataLoader.hpp"
#include <cmath>

using Catch::Approx;

TEST_CASE("DataPreprocessor - ComputeReturns", "[DataPreprocessor]") {

    MarketDataLoader::MarketData data;
    data.prices = {100.0, 105.0, 102.0};

    auto returns = DataPreprocessor::computeReturns(
        data,
        MarketDataLoader::ReturnType::LOG
    );

    REQUIRE(returns.prices.size() == 3);

    REQUIRE(returns.returns[1] == Approx(std::log(105.0 / 100.0)).margin(1e-6));
    REQUIRE(returns.returns[2] == Approx(std::log(102.0 / 105.0)).margin(1e-6));
}

TEST_CASE("DataPreprocessor - RemoveOutliers", "[DataPreprocessor]") {

    MarketDataLoader::MarketData data;
    data.prices = {1.0, 2.0, 100.0, 3.0};  // 100 est un outlier

    DataPreprocessor::removeOutliers(data, 2.0);

    REQUIRE(data.prices.size() == 3);
    REQUIRE(data.prices[2] == Approx(3.0).margin(1e-6));
}

TEST_CASE("DataPreprocessor - HandleMissingData", "[DataPreprocessor]") {

    MarketDataLoader::MarketData data;
    data.prices = {1.0, NAN, 3.0};

    DataPreprocessor::handleMissingData(data);

    // Supposé : interpolation linéaire simple
    REQUIRE(data.prices[1] == Approx(2.0).margin(1e-6));
}
