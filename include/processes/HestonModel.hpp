#pragma once
#include "../core/StochasticProcess.hpp"
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <memory>

namespace stochastic {

class HestonModel : public StochasticProcess {
private:
    // Paramètres du modèle Heston
    double mu_;      // drift
    double kappa_;   // mean reversion speed
    double theta_;   // long-term variance
    double xi_;      // vol of vol
    double rho_;     // correlation
    double v0_;      // initial variance

    double currentVariance_;  // état courant de la variance (latente)

    std::normal_distribution<double> stdNormal2_;

public:
    HestonModel(double initialPrice = 100.0,
                double mu = 0.05,
                double kappa = 2.0,
                double theta = 0.04,
                double xi = 0.3,
                double rho = -0.7,
                double v0 = 0.04,
                unsigned long seed = 0)
        : StochasticProcess(initialPrice, seed),
          mu_(mu), kappa_(kappa), theta_(theta), xi_(xi), rho_(rho), v0_(v0),
          currentVariance_(v0),
          stdNormal2_(0.0, 1.0)
    {
        validateParameters();  // On force la validité dès le départ
    }

    // === VALIDATION INCASSABLE : corrige au lieu de planter ===
    void validateParameters() const override {
        const_cast<double&>(kappa_) = std::max(1e-8, kappa_);
        const_cast<double&>(theta_) = std::max(1e-8, theta_);
        const_cast<double&>(xi_)    = std::max(1e-8, xi_);
        const_cast<double&>(v0_)    = std::max(1e-8, v0_);

        if (rho_ <= -1.0) const_cast<double&>(rho_) = -0.999;
        if (rho_ >= 1.0)  const_cast<double&>(rho_) = 0.999;

        const_cast<double&>(currentVariance_) = std::max(1e-8, currentVariance_);
    }

    double nextStep(double dt) override {
        validateParameters();  // Sécurité à chaque pas

        double S = currentState_;
        double v = currentVariance_;

        if (v < 0.0) v = 0.0;

        double Z1 = generateGaussian();
        double Z2 = rho_ * Z1 + std::sqrt(1.0 - rho_ * rho_) * stdNormal2_(generator_);

        // Prix
        double dS = mu_ * S * dt + std::sqrt(v) * S * std::sqrt(dt) * Z1;
        S += dS;

        // Variance
        double dv = kappa_ * (theta_ - v) * dt + xi_ * std::sqrt(v) * std::sqrt(dt) * Z2;
        v += dv;

        currentState_ = std::max(1e-10, S);
        currentVariance_ = std::max(0.0, v);

        return currentState_;
    }

    double drift(double, double x) const override { return mu_ * x; }
    double diffusion(double, double x) const override { return std::sqrt(currentVariance_) * x; }

    std::string getName() const override { return "Heston Model"; }

    std::vector<std::string> getParameterNames() const override {
        return {"mu", "kappa", "theta", "xi", "rho"};
    }

    std::vector<double> getParametersVector() const override {
        return {mu_, kappa_, theta_, xi_, rho_};
    }

    void setParametersVector(const std::vector<double>& params) override {
        if (params.size() != 5) {
            throw std::invalid_argument("Heston requires 5 parameters: mu, kappa, theta, xi, rho");
        }
        mu_    = params[0];
        kappa_ = params[1];
        theta_ = params[2];
        xi_    = params[3];
        rho_   = params[4];

        validateParameters();  // ON CORRIGE TOUT ICI → JAMAIS DE CRASH
    }

    double logLikelihood(const std::vector<double>& logReturns, double dt) const override {
    if (logReturns.empty()) return 0.0;

    double v = v0_;
    double ll = 0.0;

    for (size_t i = 1; i < logReturns.size(); ++i) {
        double r = logReturns[i];

        // Prédiction de la variance (Euler)
        v = v + kappa_ * (theta_ - v) * dt;
        if (v < 1e-8) v = 1e-8;

        // Vraisemblance conditionnelle (comme un GBM local)
        double sigma_t = std::sqrt(v);
        double z = (r - (mu_ - 0.5 * v) * dt) / (sigma_t * std::sqrt(dt));
        ll += -0.5 * std::log(2 * M_PI) - std::log(sigma_t * std::sqrt(dt)) - 0.5 * z * z;
    }

    return ll;
    }

    std::shared_ptr<StochasticProcess> clone() const override {
        auto copy = std::make_shared<HestonModel>(initialState_, mu_, kappa_, theta_, xi_, rho_, v0_);
        copy->setSeed(std::random_device{}());
        return copy;
    }

    void enforceConstraints() override {
        if (currentState_ <= 0.0) currentState_ = 1e-10;
        if (currentVariance_ < 0.0) currentVariance_ = 0.0;
    }
};

}