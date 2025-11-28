#include <data/MarketDataLoader.hpp>
#include <data/YahooFinanceAPI.hpp>
#include <data/DataPreprocessor.hpp>

#include <stdexcept>


MarketDataLoader::MarketData MarketDataLoader::downloadData(
    const std::string& ticker,
    const std::string& startDate,
    const std::string& endDate,
    const std::string& interval)
{
    return YahooFinanceAPI::downloadHistoricalData(ticker, startDate, endDate, interval);
}

MarketDataLoader::MarketData MarketDataLoader::computeReturns(
    const MarketData& data,
    ReturnType type) const
{
    return DataPreprocessor::computeReturns(data, type);
}

void MarketDataLoader::removeOutliers(MarketData& data, double threshold)
{
    DataPreprocessor::removeOutliers(data, threshold);
}

void MarketDataLoader::handleMissingData(MarketData& data)
{
    DataPreprocessor::handleMissingData(data);
}
