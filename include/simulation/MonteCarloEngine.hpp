/**
 * @file MonteCarloEngine.hpp
 * @brief Moteur de simulation Monte Carlo générique pour processus stochastiques.
 * @version 1.0
 */

#pragma once
#include "../core/StochasticProcess.hpp"
#include <vector>
#include <memory>
#include <future>
#include <thread>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

/**
 * @class MonteCarloEngine
 * @brief Moteur de simulation Monte Carlo supportant le parallélisme,
 *        les antithetic variates et une analyse statistique complète.
 */
class MonteCarloEngine {
public:

    /**
     * @struct SimulationConfig
     * @brief Paramètres de configuration de la simulation Monte Carlo.
     */
    struct SimulationConfig {
        size_t nPaths = 1000;             /**< Nombre de trajectoires */
        size_t nSteps = 252;              /**< Nombre de pas de temps */
        double T = 1.0;                   /**< Horizon temporel */
        bool antitheticVariates = false;  /**< Utiliser les antithetic variates */
        bool controlVariates = false;     /**< (Optionnel) Control variates */
        size_t nThreads = std::thread::hardware_concurrency(); /**< Threads utilisés */
        unsigned long seed = 42;          /**< Graine aléatoire */
        bool useExactScheme = true;       /**< Utiliser un schéma exact si disponible */
    };

    /**
     * @struct SimulationStatistics
     * @brief Statistiques détaillées des trajectoires simulées.
     */
    struct SimulationStatistics {
        std::vector<double> meanPath;     /**< Moyenne par pas de temps */
        std::vector<double> stdPath;      /**< Écart-type par pas de temps */

        std::vector<double> quant05;      /**< Quantile 5% */
        std::vector<double> quant25;      /**< Quantile 25% */
        std::vector<double> quant50;      /**< Médiane */
        std::vector<double> quant75;      /**< Quantile 75% */
        std::vector<double> quant95;      /**< Quantile 95% */

        double terminalMean{};            /**< Moyenne terminale */
        double terminalStd{};             /**< Écart-type terminal */
        double terminalMin{};             /**< Minimum terminal */
        double terminalMax{};             /**< Maximum terminal */

        double skewness{};                /**< Coefficient d’asymétrie */
        double kurtosis{};                /**< Coefficient d’aplatissement */
        double standardError{};           /**< Erreur standard */
        size_t nPaths{};                  /**< Nombre total de trajectoires */
    };

private:
    std::shared_ptr<stochastic::StochasticProcess> process_; /**< Processus simulé */
    SimulationConfig config_;                                /**< Configuration */
    std::vector<std::vector<double>> paths_;                 /**< Trajectoires */

public:

    /**
     * @brief Constructeur du moteur Monte Carlo.
     * @param process Processus stochastique à simuler.
     * @param cfg Configuration de simulation.
     */
    MonteCarloEngine(std::shared_ptr<stochastic::StochasticProcess> process,
                     const SimulationConfig& cfg);

    /**
     * @brief Lance la simulation (séquentielle ou parallèle).
     * @return Les trajectoires simulées.
     */
    std::vector<std::vector<double>> simulate();

    /**
     * @brief Renvoie l'ensemble des statistiques calculées.
     * @return Une structure contenant les statistiques.
     * @throws std::runtime_error si aucune trajectoire n'a été simulée.
     */
    SimulationStatistics getStatistics() const;

    /**
     * @brief Renvoie un indicateur de convergence (erreur standard / sigma).
     */
    double estimateConvergence() const;

    /**
     * @brief Renvoie les trajectoires simulées.
     */
    const std::vector<std::vector<double>>& getPaths() const;

    /**
     * @brief Renvoie la configuration de simulation.
     */
    SimulationConfig getConfig() const;

    /**
     * @brief Exporte les trajectoires dans un fichier CSV.
     * @param filename Chemin vers le fichier de sortie.
     */
    void exportToCSV(const std::string& filename) const;

private:

    /**
     * @brief Simulation mono-thread.
     */
    std::vector<std::vector<double>> simulateSequential();

    /**
     * @brief Simulation multi-thread (std::async).
     */
    std::vector<std::vector<double>> simulateParallel();

    /**
     * @brief Simule un lot de trajectoires (appelé par les threads).
     * @param nPaths Nombre de trajectoires à simuler.
     * @param seed Graine séparée pour éviter la corrélation.
     */
    std::vector<std::vector<double>> simulateBlock(size_t nPaths, unsigned long seed);

    /**
     * @brief Applique la technique des antithetic variates.
     */
    void applyAntitheticVariates();
};

