#include "core/TimeGrid.hpp"
#include "processes/HestonModel.hpp"
#include "processes/OrnsteinUhlenbeck.hpp"
#include "calibration/MCMCCalibrator.hpp"
#include "data/YahooFinanceAPI.hpp"

#include <iostream>
#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>

using namespace stochastic;

void printResult(const MCMCCalibrator::CalibrationResult& res) {
    std::cout << "\nRésultats calibration MCMC\n";
    std::cout << "Taux d'acceptation : " << std::fixed << std::setprecision(2)
              << res.acceptanceRate * 100 << "%\n";
    std::cout << "Log-vraisemblance  : " << res.logLikelihood << "\n";
    std::cout << "AIC / BIC          : " << res.AIC << " / " << res.BIC << "\n\n";
    for (const auto& [name, mean] : res.meanParams) {
        auto [low, high] = res.credibleIntervals95.at(name);
        std::cout << std::setw(12) << name << " = " << mean
                  << "  [" << low << ", " << high << "]\n";
    }
}

int main() {
    std::cout << "pipeline MCMC\n";

    std::string ticker;
    std::cout << "Entrez le ticker (ex: AAPL, TSLA, SPY) : ";
    std::cin >> ticker;

    std::cout << "\nChoisissez le modèle à calibrer :\n"
              << " 1. Ornstein-Uhlenbeck\n"
              << " 2. Heston\nVotre choix : ";
    int choice; std::cin >> choice;

    std::shared_ptr<StochasticProcess> process;
    MCMCCalibrator::MCMCConfig config;
    config.nIterations = 20000;
    config.burnIn = 5000;
    config.thinning = 10;
    config.adaptiveProposal = true;

    // Télécharger les données
    std::cout << "\nTéléchargement des données pour " << ticker << "...\n";
    auto data = YahooFinanceAPI::downloadHistoricalData(ticker, "2020-01-01", "2025-01-01", "1d");

    if(data.prices.size() < 100) {
        std::cerr << "Pas assez de données !\n"; return 1;
    }

    // calcul des log-returns
    std::vector<double> logReturns;
    for(size_t i=1; i<data.prices.size(); ++i)
        if(data.prices[i]>0 && data.prices[i-1]>0)
            logReturns.push_back(std::log(data.prices[i]/data.prices[i-1]));

    TimeSeriesData tsData;
    tsData.path = logReturns;
    tsData.dt = 1.0/252.0;

    // Choix du modèle et priors
    if(choice == 1) { // OU
        process = std::make_shared<OrnsteinUhlenbeck>(0.1, 0.1, 0.1, 0.1);
        config.priors["kappa"] = {MCMCCalibrator::Prior::Type::LOGNORMAL, {0.0, 0.5}};
        config.priors["theta"] = {MCMCCalibrator::Prior::Type::NORMAL, {0.0, 0.1}};
        config.priors["sigma"] = {MCMCCalibrator::Prior::Type::LOGNORMAL, {-2.0, 0.5}};
    } else if(choice == 2) { // Heston
        // Valeurs initiales sûres respectant Feller : 2*kappa*theta > xi^2
        double kappa0 = 2.0, theta0 = 0.04, xi0 = 0.1;
        process = std::make_shared<HestonModel>(100.0, 0.1, kappa0, theta0, xi0, -0.5, 0.04);

        config.priors["mu"]    = {MCMCCalibrator::Prior::Type::NORMAL, {0.0, 0.5}};
        config.priors["kappa"] = {MCMCCalibrator::Prior::Type::LOGNORMAL, {0.5, 0.5}};
        config.priors["theta"] = {MCMCCalibrator::Prior::Type::LOGNORMAL, {0.01, 0.1}};
        config.priors["xi"]    = {MCMCCalibrator::Prior::Type::LOGNORMAL, {0.01, 0.1}};
        config.priors["rho"]   = {MCMCCalibrator::Prior::Type::UNIFORM, {-0.99, 0.0}};
    } else {
        std::cerr << "Choix invalide.\n"; return 1;
    }

    // Lancer plusieurs chaînes MCMC
    size_t nChains = 4;
    std::vector<MCMCCalibrator::CalibrationResult> results;
    for(size_t c=0; c<nChains; ++c) {
        std::cout << "\n--- Chaîne " << (c+1) << " ---\n";

        auto start = std::chrono::high_resolution_clock::now();

        MCMCCalibrator calibrator(process, tsData, config);
        auto res = calibrator.calibrateFromMarketData();
        results.push_back(res);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end-start);

        std::cout << "Acceptance rate: " << res.acceptanceRate*100 << "%\n";
        std::cout << "Durée : " << duration.count() << " sec\n";
        std::cout << "Mean parameters:\n";
        for(const auto& [name,val] : res.meanParams)
            std::cout << "  " << name << " = " << val << "\n";
    }

    // Vérification convergence (simple : comparer moyennes par chaîne
    auto names = process->getParameterNames();
    for(const auto& param : names) {
        std::cout << "\nParamètre : " << param << "\n";
        for(size_t c=0; c<nChains; ++c)
            std::cout << "  Chaîne " << (c+1) << " : "
                      << results[c].meanParams.at(param) << "\n";
    }
    return 0;
}

