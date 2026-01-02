/**
 * @file MonteCarloEngine.cpp
 * @brief Implémentation du moteur Monte Carlo générique.
 */

#include "simulation/MonteCarloEngine.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

// Constructeur
/**
 * @brief Constructeur du moteur Monte Carlo.
 * @param process Processus stochastique simulé.
 * @param cfg Configuration de simulation.
 */
MonteCarloEngine::MonteCarloEngine(
    std::shared_ptr<stochastic::StochasticProcess> process,
    const SimulationConfig& cfg
)
    : process_(std::move(process)), config_(cfg)
{}

// Fonction principale de simulation
/**
 * @brief Lance la simulation Monte Carlo (choisit automatiquement parallèle ou séquentiel).
 * @return Toutes les trajectoires simulées.
 */
std::vector<std::vector<double>> MonteCarloEngine::simulate() {
    if (config_.nThreads > 1 && config_.nPaths > config_.nThreads * 4)
        return simulateParallel();
    return simulateSequential();
}

// Simulation sequentielle
/**
 * @brief Simulation mono-thread.
 */
std::vector<std::vector<double>> MonteCarloEngine::simulateSequential() {
    paths_.clear();
    paths_.reserve(config_.nPaths);

    process_->setSeed(config_.seed);

    for (size_t i = 0; i < config_.nPaths; ++i) {
        paths_.push_back(
            process_->simulatePath(config_.T, config_.nSteps, config_.useExactScheme)
        );
    }

    if (config_.antitheticVariates)
        applyAntitheticVariates();

    return paths_;
}

// Simulation parallèle
/**
 * @brief Simulation multi-thread avec std::async.
 */
std::vector<std::vector<double>> MonteCarloEngine::simulateParallel() {
    paths_.clear();
    paths_.reserve(config_.nPaths);

    const size_t block = config_.nPaths / config_.nThreads;
    const size_t rem   = config_.nPaths % config_.nThreads;

    std::vector<std::future<std::vector<std::vector<double>>>> futures;
    futures.reserve(config_.nThreads);

    for (size_t t = 0; t < config_.nThreads; ++t) {
        const size_t n = block + (t < rem ? 1 : 0);
        const unsigned long seed = config_.seed + 10000 * t;

        futures.push_back(
            std::async(std::launch::async,
                [=]() { return simulateBlock(n, seed); }
            )
        );
    }

    for (auto &f : futures) {
        auto blk = f.get();
        paths_.insert(paths_.end(), blk.begin(), blk.end());
    }

    if (config_.antitheticVariates)
        applyAntitheticVariates();

    return paths_;
}

// Simulation d'un bloc (pour le parallèlisme)
/**
 * @brief Simule un groupe de trajectoires (thread-safe).
 * @param nPaths Nombre de trajectoires.
 * @param seed Graine du thread.
 */
std::vector<std::vector<double>> MonteCarloEngine::simulateBlock(
    size_t nPaths, unsigned long seed
) {
    auto proc = process_->clone();
    proc->setSeed(seed);

    std::vector<std::vector<double>> block;
    block.reserve(nPaths);

    for (size_t i = 0; i < nPaths; ++i)
        block.push_back(
            proc->simulatePath(config_.T, config_.nSteps, config_.useExactScheme)
        );

    return block;
}

// antithetic variates
/**
 * @brief Double le nombre de trajectoires en générant des copies antithétiques.
 */
void MonteCarloEngine::applyAntitheticVariates() {
    const size_t original = paths_.size();
    paths_.reserve(original * 2);

    for (size_t i = 0; i < original; ++i)
        paths_.push_back(
            process_->simulatePath(config_.T, config_.nSteps, config_.useExactScheme)
        );
}


// statistiques
/**
 * @brief Calcule toutes les statistiques : moyennes, variances, quantiles, moments…
 */
MonteCarloEngine::SimulationStatistics MonteCarloEngine::getStatistics() const {
    if (paths_.empty())
        throw std::runtime_error("Aucune path simulée → appelez simulate() avant.");

    SimulationStatistics S;
    S.nPaths = paths_.size();

    const size_t n = paths_[0].size();
    S.meanPath.assign(n, 0.0);
    S.stdPath.assign(n, 0.0);

    S.quant05.resize(n);
    S.quant25.resize(n);
    S.quant50.resize(n);
    S.quant75.resize(n);
    S.quant95.resize(n);

    // Moyenne
    for (const auto &p : paths_)
        for (size_t i = 0; i < n; ++i)
            S.meanPath[i] += p[i];

    for (double &m : S.meanPath)
        m /= S.nPaths;

    // Écart-types + quantiles
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> v;
        v.reserve(S.nPaths);

        for (const auto &p : paths_) v.push_back(p[i]);
        std::sort(v.begin(), v.end());

        // Écart-type
        double var = 0.0;
        for (double x : v)
            var += (x - S.meanPath[i]) * (x - S.meanPath[i]);
        S.stdPath[i] = std::sqrt(var / S.nPaths);

        // Quantile helper
        auto q = [&](double a) {
            size_t idx = static_cast<size_t>(a * (v.size() - 1));
            return v[idx];
        };

        S.quant05[i] = q(0.05);
        S.quant25[i] = q(0.25);
        S.quant50[i] = q(0.50);
        S.quant75[i] = q(0.75);
        S.quant95[i] = q(0.95);
    }

    // Statistiques terminales
    std::vector<double> terminal;
    terminal.reserve(S.nPaths);
    for (const auto &p : paths_) terminal.push_back(p.back());
    std::sort(terminal.begin(), terminal.end());

    S.terminalMean = S.meanPath.back();
    S.terminalStd  = S.stdPath.back();
    S.terminalMin  = terminal.front();
    S.terminalMax  = terminal.back();

    // Moments
    double m3 = 0.0, m4 = 0.0;
    for (double x : terminal) {
        double z = (x - S.terminalMean) / S.terminalStd;
        m3 += z*z*z;
        m4 += z*z*z*z;
    }
    S.skewness = m3 / terminal.size();
    S.kurtosis = m4 / terminal.size();

    S.standardError = S.terminalStd / std::sqrt(S.nPaths);

    return S;
}

/**
 * @brief Retourne un indicateur de convergence.
 */
double MonteCarloEngine::estimateConvergence() const {
    auto S = getStatistics();
    return S.standardError / S.terminalStd;
}

// export csv
/**
 * @brief Exporte les trajectoires dans un fichier CSV.
 */
void MonteCarloEngine::exportToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Impossible d'ouvrir le fichier : " << filename << "\n";
        return;
    }

    for (const auto &path : paths_) {
        for (size_t j = 0; j < path.size(); ++j)
            file << path[j] << (j + 1 < path.size() ? ',' : '\n');
    }
}
