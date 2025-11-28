#pragma once
#include <data/MarketDataLoader.hpp>  
#include <vector>
#include <string>

class DataPreprocessor {
public:
    static MarketDataLoader::MarketData computeReturns(
        const MarketDataLoader::MarketData& input,
        MarketDataLoader::ReturnType type = MarketDataLoader::ReturnType::LOG);

    static void removeOutliers(MarketDataLoader::MarketData& data, double threshold = 3.0);

    static void handleMissingData(MarketDataLoader::MarketData& data);
};