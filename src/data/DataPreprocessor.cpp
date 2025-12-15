/**
 * @file DataPreprocessor.cpp
 * @brief Prétraitement des données financières de marché.
 *
 * Ce fichier regroupe les fonctions utilitaires permettant
 * de préparer des séries temporelles financières avant
 * calibration ou simulation :
 *
 *  - calcul des rendements (logarithmiques ou simples)
 *  - suppression d’outliers de type spike
 *  - gestion des données manquantes par interpolation
 *
 * Ces étapes sont cruciales pour garantir la stabilité
 * numérique des estimateurs statistiques et des méthodes
 * de calibration probabilistes.
 *
 * @author —
 * @date 2025
 */

#include <data/DataPreprocessor.hpp>
#include <data/MarketDataLoader.hpp>

#include <numeric>
#include <cmath>
#include <iostream>

// ============================================================================
// CALCUL DES RENDEMENTS
// ============================================================================

/**
 * @brief Calcule les rendements à partir des prix.
 *
 * Les rendements sont alignés avec les prix :
 * - même taille que le vecteur des prix
 * - premier rendement fixé à 0.0
 *
 * @param input Données de marché (prix + dates)
 * @param type Type de rendement (LOG ou SIMPLE)
 * @return Données enrichies avec les rendements
 */
MarketDataLoader::MarketData
DataPreprocessor::computeReturns(
    const MarketDataLoader::MarketData& input,
    MarketDataLoader::ReturnType type
) {
    MarketDataLoader::MarketData result = input;

    // Alignement rendements / prix
    result.returns.clear();
    result.returns.assign(input.prices.size(), 0.0);

    if (input.prices.size() < 2)
        return result;

    for (size_t i = 1; i < input.prices.size(); ++i) {

        // Sécurité numérique
        if (input.prices[i] <= 0.0 || input.prices[i-1] <= 0.0) {
            result.returns[i] = 0.0;
            continue;
        }

        if (type == MarketDataLoader::ReturnType::LOG) {
            result.returns[i] =
                std::log(input.prices[i] / input.prices[i-1]);
        } else {
            result.returns[i] =
                (input.prices[i] - input.prices[i-1])
                / input.prices[i-1];
        }
    }

    return result;
}

// ============================================================================
// SUPPRESSION DES OUTLIERS
// ============================================================================

/**
 * @brief Supprime les outliers de type spike isolé.
 *
 * Un point est considéré comme un outlier si :
 *  - le rendement précédent est élevé en valeur absolue
 *  - le rendement suivant est également élevé
 *  - les deux rendements sont de signes opposés
 *
 * Cela correspond typiquement à des erreurs de cotation
 * ou à des artefacts de données (ex : 2 → 100 → 3).
 *
 * @param data Données de marché (modifiées en place)
 * @param threshold Seuil absolu sur les rendements
 */
void DataPreprocessor::removeOutliers(
    MarketDataLoader::MarketData& data,
    double threshold
) {
    if (data.prices.size() < 3)
        return;

    // Calcul des rendements si nécessaire
    if (data.returns.empty()) {
        data = computeReturns(
            data,
            MarketDataLoader::ReturnType::LOG
        );
    }

    // Statistiques de base (informatives)
    double mean =
        std::accumulate(
            data.returns.begin(),
            data.returns.end(),
            0.0
        ) / data.returns.size();

    double variance = 0.0;
    for (double r : data.returns)
        variance += (r - mean) * (r - mean);

    variance /= data.returns.size();
    double stddev = std::sqrt(variance);

    if (stddev == 0.0)
        return;

    std::vector<double> cleanPrices;
    std::vector<std::string> cleanDates;

    bool hasDates =
        (data.dates.size() == data.prices.size());

    // Conservation du premier point
    cleanPrices.push_back(data.prices[0]);
    if (hasDates)
        cleanDates.push_back(data.dates[0]);

    size_t n = data.prices.size();
    std::vector<bool> toRemove(n, false);

    // Détection des spikes centraux
    for (size_t i = 1; i + 1 < n; ++i) {
        double r_prev = data.returns[i];
        double r_next = data.returns[i+1];

        if (std::abs(r_prev) > threshold &&
            std::abs(r_next) > threshold &&
            (r_prev * r_next < 0)) {

            toRemove[i] = true;
        }
    }

    // Reconstruction des séries nettoyées
    for (size_t i = 1; i < n; ++i) {
        if (!toRemove[i]) {
            cleanPrices.push_back(data.prices[i]);
            if (hasDates)
                cleanDates.push_back(data.dates[i]);
        }
    }

    data.prices = std::move(cleanPrices);
    if (hasDates)
        data.dates = std::move(cleanDates);

    // Recalcul propre des rendements
    data.returns.clear();
    data = computeReturns(
        data,
        MarketDataLoader::ReturnType::LOG
    );
}

// ============================================================================
// DONNÉES MANQUANTES
// ============================================================================

/**
 * @brief Gère les données de prix manquantes.
 *
 * Les valeurs NaN sont remplacées par interpolation
 * linéaire entre les observations valides adjacentes.
 *
 * @param data Données de marché (modifiées en place)
 */
void DataPreprocessor::handleMissingData(
    MarketDataLoader::MarketData& data
) {
    for (size_t i = 1; i < data.prices.size(); ++i) {

        if (std::isnan(data.prices[i])) {

            // Recherche du prochain point valide
            size_t j = i + 1;
            while (j < data.prices.size() &&
                   std::isnan(data.prices[j])) {
                ++j;
            }

            double start = data.prices[i - 1];
            double end =
                (j < data.prices.size())
                ? data.prices[j]
                : start;

            // Interpolation linéaire
            for (size_t k = i; k < j; ++k) {
                data.prices[k] =
                    start +
                    (end - start) *
                    (k - i + 1) /
                    (j - i + 1);
            }

            i = j - 1;
        }
    }

    // Recalcul des rendements après correction
    data = computeReturns(
        data,
        MarketDataLoader::ReturnType::LOG
    );
}
