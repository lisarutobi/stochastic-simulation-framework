#pragma once
#include <string>
#include <vector>

class MarketDataLoader {
public:
    enum class ReturnType { LOG, SIMPLE };

    struct MarketData {
        std::vector<double> prices;
        std::vector<double> returns;
        std::vector<std::string> dates;
        std::string ticker;
        double samplingFrequency = 252.0;
    };

    MarketDataLoader() = default;

    MarketData downloadData(const std::string& ticker,
                            const std::string& startDate,
                            const std::string& endDate,
                            const std::string& interval
                            );

    MarketData computeReturns(const MarketData& data,
                              ReturnType type = ReturnType::LOG) const;

    void removeOutliers(MarketData& data, double threshold = 3.0);
    void handleMissingData(MarketData& data);

private:
};
