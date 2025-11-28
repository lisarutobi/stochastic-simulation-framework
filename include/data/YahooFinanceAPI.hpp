#pragma once

#include "data/MarketDataLoader.hpp"
#include <string>

class YahooFinanceAPI {
public:
    static MarketDataLoader::MarketData downloadHistoricalData(
        const std::string& ticker,
        const std::string& startDate,    // format YYYY-MM-DD
        const std::string& endDate,      // format YYYY-MM-DD
        const std::string& interval = "1d"      //(1d, 1h, 1wk, 1mo)
    );
};
