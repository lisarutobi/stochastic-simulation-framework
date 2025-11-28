#include "core/TimeGrid.hpp"
#include "processes/GeometricBrownianMotion.hpp"
#include "processes/HestonModel.hpp"
#include "processes/OrnsteinUhlenbeck.hpp"
#include "processes/CoxIngersollRoss.hpp"
#include "processes/LevyJumpDiffusion.hpp"
#include "processes/FractionalBrownianMotion.hpp"

#include "data/YahooFinanceAPI.hpp"
#include "calibration/MCMCCalibrator.hpp"

#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>

using namespace stochastic;

void printResult(const MCMCCalibrator::CalibrationResult& res) {
    std::cout << "\n=== RÉSULTATS DE CALIBRATION MCMC ===\n";
    std::cout << "Taux d'acceptation : " << std::fixed << std::setprecision(3) << res.acceptanceRate * 100 << "%\n";
    std::cout << "Log-vraisemblance  : " << res.logLikelihood << "\n";
    std::cout << "AIC / BIC          : " << res.AIC << " / " << res.BIC << "\n\n";

    std::cout << "Paramètres estimés (moyenne ± IC 95%) :\n";
    for (const auto& [name, mean] : res.meanParams) {
        auto [low, high] = res.credibleIntervals95.at(name);
        std::cout << "  " << std::setw(12) << name << " = "
                  << std::fixed << std::setprecision(6) << mean
                  << "  [" << low << ", " << high << "]\n";
    }
}

int main() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════╗
║     FRAMEWORK DE SIMULATION STOCHASTIQUE 2025            ║
║     Calibration MCMC + Yahoo Finance + 7 modèles         ║
╚═══════════════════════════════════════════════════════════╝
)" << std::endl;

    std::string ticker;
    std::cout << "Entrez le ticker (ex: AAPL, TSLA, SPY, BTC-USD) : ";
    std::cin >> ticker;

    std::cout << "\nTéléchargement des données pour " << ticker << " ...\n";
    auto data = YahooFinanceAPI::downloadHistoricalData(ticker, "2020-01-01", "2025-01-01", "1d");

    if (data.prices.size() < 100) {
        std::cerr << "Pas assez de données !\n";
        return 1;
    }

    // Calcul des log-returns
    std::vector<double> logReturns;
    logReturns.reserve(data.prices.size() - 1);
    for (size_t i = 1; i < data.prices.size(); ++i) {
        if (data.prices[i] > 0 && data.prices[i-1] > 0) {
            logReturns.push_back(std::log(data.prices[i] / data.prices[i-1]));
        }
    }

    TimeSeriesData tsData;
    tsData.path = logReturns;
    tsData.dt = 1.0 / 252.0;

    std::cout << "\n" << logReturns.size() << " log-returns chargés (dt = 1/252 an)\n";

    std::cout << R"(
Choisissez le modèle à calibrer :
  1. Geometric Brownian Motion
  2. Heston Model
  3. Ornstein-Uhlenbeck (sur log-returns)
  4. Levy Jump-Diffusion
  5. Fractional Brownian Motion
Votre choix (1-5) : )";

    int choice;
    std::cin >> choice;

    std::shared_ptr<StochasticProcess> process;
    MCMCCalibrator::MCMCConfig config;
    config.nIterations = 30000;
    config.burnIn = 5000;
    config.thinning = 10;
    config.adaptiveProposal = true;

    switch(choice) {
        case 1: { // GBM
            process = std::make_shared<GeometricBrownianMotion>(0.0, 0.05, 0.2);
            config.priors["mu"]    = {MCMCCalibrator::Prior::Type::NORMAL,    {0.0, 0.2}};
            config.priors["sigma"] = {MCMCCalibrator::Prior::Type::LOGNORMAL, { -2.0, 0.5}};
            break;
        }

        case 2: {
            process = std::make_shared<HestonModel>(100.0, 0.1, 2.0, 0.04, 0.4, -0.7, 0.04);

            config.priors["mu"]    = {MCMCCalibrator::Prior::Type::NORMAL,    {0.1, 0.5}};
            config.priors["kappa"] = {MCMCCalibrator::Prior::Type::NORMAL,    {2.0, 3.0}};
            config.priors["theta"] = {MCMCCalibrator::Prior::Type::LOGNORMAL, {-3.2, 0.6}};
            config.priors["xi"]    = {MCMCCalibrator::Prior::Type::LOGNORMAL, {-0.5, 1.0}};
            config.priors["rho"]   = {MCMCCalibrator::Prior::Type::UNIFORM,   {-0.99, 0.1}};

            process->setParametersVector({0.15, 2.5, 0.045, 0.5, -0.8});
            break;
        }

        case 3: { // OU
            process = std::make_shared<OrnsteinUhlenbeck>(0.0, 1.5, 0.0, 0.1);
            config.priors["kappa"] = {MCMCCalibrator::Prior::Type::NORMAL, {1.0, 1.0}};
            config.priors["theta"] = {MCMCCalibrator::Prior::Type::NORMAL, {0.0, 0.1}};
            config.priors["sigma"] = {MCMCCalibrator::Prior::Type::LOGNORMAL, {-2.5, 0.6}};
            break;
        }
        case 4: { // Levy
            process = std::make_shared<LevyJumpDiffusion>(0.0, 0.05, 0.2, 1.0, -0.1, 0.15);
            config.priors["lambda"] = {MCMCCalibrator::Prior::Type::UNIFORM, {0.1, 10.0}};
            config.priors["mu_j"]   = {MCMCCalibrator::Prior::Type::NORMAL, {-0.1, 0.2}};
            config.priors["sigma_j"]= {MCMCCalibrator::Prior::Type::LOGNORMAL, {-2.0, 0.8}};
            break;
        }
        case 5: { // fBM
            process = std::make_shared<FractionalBrownianMotion>(0.0, 0.6, 0.02);
            config.priors["H"]     = {MCMCCalibrator::Prior::Type::UNIFORM, {0.05, 0.95}};
            config.priors["sigma"] = {MCMCCalibrator::Prior::Type::LOGNORMAL, {-4.0, 1.0}};
            break;
        }
        default:
            std::cout << "Choix invalide.\n"; return 1;
    }

    std::cout << "\nLancement de la calibration MCMC sur " << ticker << " avec " << process->getName() << "...\n";
    auto start = std::chrono::high_resolution_clock::now();

    MCMCCalibrator calibrator(process, tsData, config);
    auto result = calibrator.calibrateFromMarketData();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    printResult(result);
    std::cout << "\nCalibration terminée en " << duration.count() << " secondes.\n";

    return 0;
}