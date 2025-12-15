/**
 * @file YahooFinanceAPI.cpp
 * @brief Téléchargement de données financières via Yahoo Finance.
 *
 * Ce fichier implémente une interface bas niveau
 * utilisant libcurl pour interroger l’API non officielle
 * de Yahoo Finance et parser les réponses JSON.
 */

#include "data/YahooFinanceAPI.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <ctime>
#include <stdexcept>
#include <cstdio>
#include <iostream>

using json = nlohmann::json;

// ============================================================================
// FONCTIONS INTERNES
// ============================================================================

namespace {

/**
 * @brief Callback CURL pour accumuler la réponse HTTP.
 */
size_t WriteCallback(
    void* contents,
    size_t size,
    size_t nmemb,
    void* userp
) {
    size_t realsize = size * nmemb;
    auto& buffer = *static_cast<std::string*>(userp);
    buffer.append(static_cast<char*>(contents), realsize);
    return realsize;
}

/**
 * @brief Convertit une date YYYY-MM-DD en timestamp UNIX.
 */
uint64_t dateToUnixTimestamp(const std::string& date) {
    int y, m, d;
    if (std::sscanf(date.c_str(), "%d-%d-%d", &y, &m, &d) != 3)
        throw std::invalid_argument(
            "Invalid date format, expected YYYY-MM-DD"
        );

    if (m < 1 || m > 12)
        throw std::invalid_argument("Month out of range");
    if (d < 1 || d > 31)
        throw std::invalid_argument("Day out of range");

    std::tm timeinfo = {};
    timeinfo.tm_year = y - 1900;
    timeinfo.tm_mon  = m - 1;
    timeinfo.tm_mday = d;
    timeinfo.tm_isdst = -1;

    return static_cast<uint64_t>(std::mktime(&timeinfo));
}

} // namespace

// ============================================================================
// API PRINCIPALE
// ============================================================================

/**
 * @brief Télécharge les données historiques d’un actif.
 *
 * @param ticker Symbole Yahoo Finance
 * @param startDate Date de début
 * @param endDate Date de fin
 * @param interval Fréquence
 * @return Données de marché
 */
MarketDataLoader::MarketData
YahooFinanceAPI::downloadHistoricalData(
    const std::string& ticker,
    const std::string& startDate,
    const std::string& endDate,
    const std::string& interval
) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* curl = curl_easy_init();
    if (!curl)
        throw std::runtime_error("Failed to initialize CURL");

    std::string buffer;

    uint64_t start_ts = dateToUnixTimestamp(startDate);
    uint64_t end_ts   = dateToUnixTimestamp(endDate) + 86400;

    std::string url =
        "https://query1.finance.yahoo.com/v8/finance/chart/" +
        ticker +
        "?interval=" + interval +
        "&period1=" + std::to_string(start_ts) +
        "&period2=" + std::to_string(end_ts);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    if (res != CURLE_OK)
        throw std::runtime_error(
            std::string("CURL error: ") +
            curl_easy_strerror(res)
        );

    json j = json::parse(buffer);

    if (j["chart"]["error"] != nullptr &&
        !j["chart"]["error"].is_null()) {
        throw std::runtime_error(
            "Yahoo Finance API returned an error for ticker " + ticker
        );
    }

    auto result = j["chart"]["result"][0];
    auto timestamps = result["timestamp"];
    auto closes =
        result["indicators"]["quote"][0]["close"];

    MarketDataLoader::MarketData data;
    data.ticker = ticker;
    data.samplingFrequency = 252.0;

    for (size_t i = 0; i < timestamps.size(); ++i) {
        if (closes[i].is_null())
            continue;

        double price = closes[i].get<double>();
        if (price <= 0.0)
            continue;

        std::time_t ts = timestamps[i];
        char buf[11];
        std::strftime(buf, sizeof(buf), "%Y-%m-%d",
                      std::gmtime(&ts));

        data.dates.push_back(buf);
        data.prices.push_back(price);
    }

    if (data.prices.empty())
        throw std::runtime_error(
            "No price data returned for ticker " + ticker
        );

    return data;
}
