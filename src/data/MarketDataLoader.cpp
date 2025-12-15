/**
 * @file MarketDataLoader.cpp
 * @brief Chargement et prétraitement des données de marché.
 *
 * Ce fichier implémente une interface simple pour :
 *  - télécharger des données historiques
 *  - calculer les rendements
 *  - nettoyer les données (outliers, valeurs manquantes)
 *
 * Il agit comme une façade autour de YahooFinanceAPI
 * et de DataPreprocessor.
 *
 * @author —
 * @date 2025
 */

#include <data/MarketDataLoader.hpp>
#include <data/YahooFinanceAPI.hpp>
#include <data/DataPreprocessor.hpp>

#include <stdexcept>

// ============================================================================
// TÉLÉCHARGEMENT DES DONNÉES
// ============================================================================

/**
 * @brief Télécharge des données historiques depuis Yahoo Finance.
 *
 * @param ticker Symbole du sous-jacent (ex: AAPL, EURUSD=X)
 * @param startDate Date de début (YYYY-MM-DD)
 * @param endDate Date de fin (YYYY-MM-DD)
 * @param interval Fréquence d’échantillonnage (1d, 1h, etc.)
 * @return Données de marché brutes
 */
MarketDataLoader::MarketData
MarketDataLoader::downloadData(
    const std::string& ticker,
    const std::string& startDate,
    const std::string& endDate,
    const std::string& interval
) {
    return YahooFinanceAPI::downloadHistoricalData(
        ticker, startDate, endDate, interval
    );
}

// ============================================================================
// CALCUL DES RENDEMENTS
// ============================================================================

/**
 * @brief Calcule les rendements à partir des prix.
 *
 * @param data Données de marché
 * @param type Type de rendement (LOG ou SIMPLE)
 * @return Données enrichies avec rendements
 */
MarketDataLoader::MarketData
MarketDataLoader::computeReturns(
    const MarketData& data,
    ReturnType type
) const {
    return DataPreprocessor::computeReturns(data, type);
}

// ============================================================================
// NETTOYAGE DES DONNÉES
// ============================================================================

/**
 * @brief Supprime les outliers de type spike.
 *
 * @param data Données de marché (modifiées en place)
 * @param threshold Seuil absolu sur les rendements
 */
void MarketDataLoader::removeOutliers(
    MarketData& data,
    double threshold
) {
    DataPreprocessor::removeOutliers(data, threshold);
}

/**
 * @brief Traite les données manquantes par interpolation.
 *
 * @param data Données de marché (modifiées en place)
 */
void MarketDataLoader::handleMissingData(
    MarketData& data
) {
    DataPreprocessor::handleMissingData(data);
}
