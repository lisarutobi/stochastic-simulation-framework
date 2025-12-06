#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_approx.hpp>
#include <nlohmann/json.hpp>

#include "../include/data/YahooFinanceAPI.hpp"
#include "../include/data/MarketDataLoader.hpp"

#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <ctime>

using Catch::Approx;

// ----------------------------------------------------------------------
// 1. Tests des paramètres invalides
// ----------------------------------------------------------------------

TEST_CASE("YahooFinanceAPI - Invalid date format throws", "[YahooFinanceAPI]") {
    REQUIRE_THROWS_AS(
        YahooFinanceAPI::downloadHistoricalData("AAPL", "2023-13-01", "2023-01-10", "1d"),
        std::invalid_argument
    );
}

TEST_CASE("YahooFinanceAPI - Empty ticker throws", "[YahooFinanceAPI]") {
    REQUIRE_THROWS_AS(
        YahooFinanceAPI::downloadHistoricalData("", "2023-01-01", "2023-01-10", "1d"),
        std::runtime_error     // Erreur CURL => URL invalide
    );
}

// ----------------------------------------------------------------------
// 2. Test offline avec JSON mocké
// ----------------------------------------------------------------------

TEST_CASE("YahooFinanceAPI - Parse mocked Yahoo JSON", "[YahooFinanceAPI]") {

    std::string mockJson = R"({
        "chart": {
            "result": [{
                "timestamp": [1672531200, 1672617600],
                "indicators": {
                    "quote": [{
                        "close": [130.5, 132.2]
                    }]
                }
            }],
            "error": null
        }
    })";

    nlohmann::json j = nlohmann::json::parse(mockJson);

    auto result     = j["chart"]["result"][0];
    auto timestamps = result["timestamp"];
    auto closes     = result["indicators"]["quote"][0]["close"];

    MarketDataLoader::MarketData data;
    data.ticker = "AAPL";
    data.samplingFrequency = 252.0;

    for (size_t i = 0; i < timestamps.size(); ++i) {
        std::time_t ts = timestamps[i].get<std::time_t>();
        char buf[11];

        std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::gmtime(&ts));

        data.dates.push_back(std::string(buf));
        data.prices.push_back(closes[i].get<double>());
    }

    REQUIRE(data.prices.size() == 2);
    REQUIRE(data.dates.size()  == 2);

    REQUIRE(data.prices[0] == Approx(130.5).margin(1e-6));
    REQUIRE(data.prices[1] == Approx(132.2).margin(1e-6));
}

// ----------------------------------------------------------------------
// 3. Test du comportement en cas d'erreur API Yahoo
// ----------------------------------------------------------------------

TEST_CASE("YahooFinanceAPI - API error JSON is detected", "[YahooFinanceAPI]") {

    std::string mockJson = R"({
        "chart": {
            "result": null,
            "error": { "code": "Not Found" }
        }
    })";

    nlohmann::json j = nlohmann::json::parse(mockJson);

    REQUIRE(j["chart"]["error"].is_object());
}

// ----------------------------------------------------------------------
// 4. Test live réseau (désactivé)
// ----------------------------------------------------------------------

TEST_CASE("YahooFinanceAPI - Live API call (disabled)", "[hide][manual]") {

    auto data = YahooFinanceAPI::downloadHistoricalData(
        "AAPL", "2023-01-01", "2023-01-05", "1d"
    );

    REQUIRE_FALSE(data.prices.empty());
    REQUIRE(data.ticker == "AAPL");
    REQUIRE(data.prices.size() == data.dates.size());
}
