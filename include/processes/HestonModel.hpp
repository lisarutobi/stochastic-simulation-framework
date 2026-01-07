#pragma once

#include "core/StochasticProcess.hpp"
#include "core/ProcessParameters.hpp"
#include <cmath>
#include <limits>

namespace stochastic {

class HestonModel : public StochasticProcess {
private:
    HestonParameters* hestonParams_;
    double currentVariance_;

public:
    explicit HestonModel(
        double initialPrice = 100.0,
        double mu = 0.05,
        double kappa = 2.0,
        double theta = 0.04,
        double xi = 0.3,
        double rho = -0.7,
        double v0 = 0.04,
        unsigned long seed = 0
    ) : StochasticProcess(initialPrice, seed), currentVariance_(v0) {
        auto params = std::make_unique<HestonParameters>(mu, kappa, theta, xi, rho, v0);
        params->validate();
        hestonParams_ = params.get();
        params_ = std::move(params);
    }

    HestonModel(
        double initialPrice,
        std::unique_ptr<HestonParameters> params,
        unsigned long seed = 0
    ) : StochasticProcess(initialPrice, seed) {
        params->validate();
        hestonParams_ = params.get();
        currentVariance_ = hestonParams_->getV0();
        params_ = std::move(params);
    }

    double nextStep(double dt) override {
        double S = currentState_;
        double v = currentVariance_;

        double mu = hestonParams_->getMu();
        double kappa = hestonParams_->getKappa();
        double theta = hestonParams_->getTheta();
        double xi = hestonParams_->getXi();
        double rho = hestonParams_->getRho();

        if (v < 0.0) v = 0.0;

        double Z1 = generateGaussian();
        double Z2 = rho * Z1 + std::sqrt(1.0 - rho * rho) * generateGaussian();

        // Dynamique du prix
        double dS = mu * S * dt + std::sqrt(v) * S * std::sqrt(dt) * Z1;
        S += dS;

        // Dynamique de la variance
        double dv = kappa * (theta - v) * dt + xi * std::sqrt(v) * std::sqrt(dt) * Z2;
        v += dv;

        currentState_ = std::max(1e-10, S);
        currentVariance_ = std::max(0.0, v);

        return currentState_;
    }

    double drift(double t, double x) const override {
        (void)t;
        return hestonParams_->getMu() * x;
    }

    double diffusion(double t, double x) const override {
        (void)t;
        return std::sqrt(currentVariance_) * x;
    }

    std::string getName() const override {
        return "Heston Model";
    }

    void validateParameters() const override {
        hestonParams_->validate();
    }

    bool hasStationaryDistribution() const override {
        return false;
    }

    bool canBeNegative() const override {
        return false;
    }

    void enforceConstraints() override {
        if (currentState_ <= 0.0) currentState_ = 1e-10;
        if (currentVariance_ < 0.0) currentVariance_ = 0.0;
    }

    std::vector<double> simulatePathExact(const TimeGrid& timeGrid) override {
        reset();
        currentVariance_ = hestonParams_->getV0();

        std::vector<double> path;
        path.reserve(timeGrid.size());
        path.push_back(currentState_);

        for (size_t i = 0; i < timeGrid.getNumSteps(); ++i) {
            double dt = timeGrid.getIncrement(i);
            currentState_ = nextStep(dt);
            path.push_back(currentState_);
        }
        return path;
    }

    std::vector<std::string> getParameterNames() const override {
        return {"mu", "kappa", "theta", "xi", "rho"};
    }

    std::vector<double> getParametersVector() const override {
        return {
            hestonParams_->getMu(),
            hestonParams_->getKappa(),
            hestonParams_->getTheta(),
            hestonParams_->getXi(),
            hestonParams_->getRho()
        };
    }

    void setParametersVector(const std::vector<double>& params) override {
        if (params.size() != 5) {
            throw std::invalid_argument("Heston requires 5 parameters: mu, kappa, theta, xi, rho");
        }
        auto newParams = std::make_unique<HestonParameters>(
            params[0], params[1], params[2], params[3], params[4], hestonParams_->getV0()
        );
        newParams->validate();
        hestonParams_ = newParams.get();
        params_ = std::move(newParams);
    }

    double logLikelihood(const std::vector<double>& logReturns, double dt) const override {
        if (logReturns.empty()) return 0.0;

        double mu = hestonParams_->getMu();
        double kappa = hestonParams_->getKappa();
        double theta = hestonParams_->getTheta();

        double v = hestonParams_->getV0();
        double ll = 0.0;

        for (size_t i = 1; i < logReturns.size(); ++i) {
            double r = logReturns[i];

            //prédiction Euler de la variance
            v = v + kappa * (theta - v) * dt;
            if (v < 1e-8) v = 1e-8;

            //vraisemblance conditionnelle gaussienne
            double sigma_t = std::sqrt(v);
            double expectedReturn = (mu - 0.5 * v) * dt;
            double variance = v * dt;

            if (variance <= 0) return -std::numeric_limits<double>::infinity();

            double z = (r - expectedReturn) / (sigma_t * std::sqrt(dt));
            ll += -0.5 * std::log(2.0 * M_PI * variance) - 0.5 * z * z;
        }

        return ll;
    }

    std::shared_ptr<StochasticProcess> clone() const override {
        auto paramsClone = std::make_unique<HestonParameters>(*hestonParams_);
        return std::make_shared<HestonModel>(initialState_, std::move(paramsClone), 0);
    }

    // Accesseurs spécifiques
    double getCurrentVariance() const { return currentVariance_; }
    void setCurrentVariance(double v) { currentVariance_ = std::max(0.0, v); }

    void reset() {
        StochasticProcess::reset();
        currentVariance_ = hestonParams_->getV0();
    }
};

}
