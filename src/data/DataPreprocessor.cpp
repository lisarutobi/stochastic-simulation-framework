#include <data/DataPreprocessor.hpp>
#include <data/MarketDataLoader.hpp>  

#include <numeric>
#include <cmath>
#include <iostream>

MarketDataLoader::MarketData DataPreprocessor::computeReturns(
    const MarketDataLoader::MarketData& input,
    MarketDataLoader::ReturnType type)
{
    MarketDataLoader::MarketData result = input;
    result.returns.clear();

    if (input.prices.size() < 2) return result;

    result.returns.reserve(input.prices.size() - 1);

    for (size_t i = 1; i < input.prices.size(); ++i) {
        if (input.prices[i] <= 0.0 || input.prices[i-1] <= 0.0) {
            result.returns.push_back(0.0);
            continue;
        }

        if (type == MarketDataLoader::ReturnType::LOG) {
            result.returns.push_back(std::log(input.prices[i] / input.prices[i-1]));
        } else {
            result.returns.push_back((input.prices[i] - input.prices[i-1]) / input.prices[i-1]);
        }
    }
    return result;
}

void DataPreprocessor::removeOutliers(MarketDataLoader::MarketData& data, double threshold) {

    if (data.prices.size() < 10) return;

    // On calcule les returns
    if (data.returns.empty()) {
        data = computeReturns(data, MarketDataLoader::ReturnType::LOG);
    }

    double mean = std::accumulate(data.returns.begin(), data.returns.end(), 0.0) / data.returns.size();

    double variance = 0.0;
    for (double r : data.returns) variance += (r - mean) * (r - mean);

    variance /= data.returns.size();
    double stddev = std::sqrt(variance);

    if (stddev == 0.0) return;

    std::vector<double> cleanPrices;
    std::vector<std::string> cleanDates;

    cleanPrices.push_back(data.prices[0]);
    cleanDates.push_back(data.dates[0]);

    for (size_t i = 1; i < data.prices.size(); ++i) {
        double r = data.returns[i - 1];
        if (std::abs(r - mean) <= threshold * stddev) {
            cleanPrices.push_back(data.prices[i]);
            cleanDates.push_back(data.dates[i]);
        }
    }

    data.prices = std::move(cleanPrices);
    data.dates = std::move(cleanDates);
    data.returns.clear();
    data = computeReturns(data, MarketDataLoader::ReturnType::LOG); // on recalcule proprement
}

void DataPreprocessor::handleMissingData(MarketDataLoader::MarketData& data) {

    // Forward fill simple
    for (size_t i = 1; i < data.prices.size(); ++i) {
        if (data.prices[i] <= 0.0 || std::isnan(data.prices[i])) {
            data.prices[i] = data.prices[i - 1];
        }
    }

    // On recalcule les returns après fill
    data = computeReturns(data, MarketDataLoader::ReturnType::LOG);
}