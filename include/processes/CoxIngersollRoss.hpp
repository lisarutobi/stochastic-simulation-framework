#pragma once

#include "core/StochasticProcess.hpp"
#include "core/ProcessParameters.hpp"
#include <cmath>
#include <limits>

namespace stochastic {

class CoxIngersollRoss : public StochasticProcess {
private:
    CIRParameters* cirParams_;
    mutable std::gamma_distribution<double> gammaDist_;

    // Échantillonnage chi² non-central via mélange Poisson-Gamma
    double sampleNonCentralChiSquared(double df, double ncp) {
        if (ncp <= 0) {
            std::gamma_distribution<double> gamma(df / 2.0, 2.0);
            return gamma(generator_);
        }

        std::poisson_distribution<int> poisson(ncp / 2.0);
        int N = poisson(generator_);

        double newDf = df + 2.0 * N;
        if (newDf <= 0) newDf = 0.001;

        std::gamma_distribution<double> gamma(newDf / 2.0, 2.0);
        return gamma(generator_);
    }

    // Log de la fonction de Bessel modifiée I_v(x)
    static double logBesselI(double v, double x) {
        if (x <= 0) return -std::numeric_limits<double>::infinity();

        if (x < 30.0) {
            double sum = 0.0;
            double logTerm = v * std::log(x / 2.0) - std::lgamma(v + 1.0);
            sum = std::exp(logTerm);

            for (int k = 1; k < 100; ++k) {
                logTerm = (v + 2.0 * k) * std::log(x / 2.0)
                        - std::lgamma(k + 1.0)
                        - std::lgamma(v + k + 1.0);
                double term = std::exp(logTerm);
                sum += term;
                if (term < sum * 1e-15) break;
            }
            return std::log(sum);
        } else {
            return x - 0.5 * std::log(2.0 * M_PI * x)
                   - (4.0 * v * v - 1.0) / (8.0 * x);
        }
    }

public:
    explicit CoxIngersollRoss(
        double initialRate = 0.03,
        double kappa = 0.5,
        double theta = 0.03,
        double sigma = 0.1,
        unsigned long seed = 0
    ) : StochasticProcess(initialRate, seed), gammaDist_(1.0, 1.0) {
        auto params = std::make_unique<CIRParameters>(kappa, theta, sigma);
        params->validate();
        cirParams_ = params.get();
        params_ = std::move(params);
    }

    CoxIngersollRoss(
        double initialRate,
        std::unique_ptr<CIRParameters> params,
        unsigned long seed = 0
    ) : StochasticProcess(initialRate, seed), gammaDist_(1.0, 1.0) {
        params->validate();
        cirParams_ = params.get();
        params_ = std::move(params);
    }

    double nextStep(double dt) override {
        double r = currentState_;
        double kappa = cirParams_->getKappa();
        double theta = cirParams_->getTheta();
        double sigma = cirParams_->getSigma();

        double dr = kappa * (theta - r) * dt
                  + sigma * std::sqrt(std::max(r, 0.0) * dt) * generateGaussian();
        r += dr;

        currentState_ = r;
        enforceConstraints();
        return r;
    }

    double drift(double t, double x) const override {
        (void)t;
        return cirParams_->getKappa() * (cirParams_->getTheta() - x);
    }

    double diffusion(double t, double x) const override {
        (void)t;
        return cirParams_->getSigma() * std::sqrt(std::max(x, 0.0));
    }

    std::string getName() const override {
        return "Cox-Ingersoll-Ross (CIR)";
    }

    void validateParameters() const override {
        cirParams_->validate();
    }

    bool hasStationaryDistribution() const override {
        return true;
    }

    bool canBeNegative() const override {
        return false;
    }

    void enforceConstraints() override {
        if (currentState_ < 0.0) {
            currentState_ = 0.0;
        }
    }

    // Simulation exacte via distribution chi² non-centrale
    std::vector<double> simulatePathExact(const TimeGrid& timeGrid) override {
        reset();
        std::vector<double> path;
        path.reserve(timeGrid.size());
        path.push_back(currentState_);

        double kappa = cirParams_->getKappa();
        double theta = cirParams_->getTheta();
        double sigma = cirParams_->getSigma();

        double prev_r = currentState_;

        for (size_t i = 0; i < timeGrid.getNumSteps(); ++i) {
            double dt = timeGrid.getIncrement(i);

            double expKappaDt = std::exp(-kappa * dt);
            double c = sigma * sigma * (1.0 - expKappaDt) / (4.0 * kappa);
            double d = 4.0 * kappa * theta / (sigma * sigma);
            double lambda = 4.0 * kappa * prev_r * expKappaDt
                          / (sigma * sigma * (1.0 - expKappaDt));

            double chi2 = sampleNonCentralChiSquared(d, lambda);
            double r_next = c * chi2;

            currentState_ = std::max(r_next, 0.0);
            path.push_back(currentState_);
            prev_r = currentState_;
        }
        return path;
    }

    std::vector<std::string> getParameterNames() const override {
        return {"kappa", "theta", "sigma"};
    }

    std::vector<double> getParametersVector() const override {
        return {cirParams_->getKappa(), cirParams_->getTheta(), cirParams_->getSigma()};
    }

    void setParametersVector(const std::vector<double>& params) override {
        if (params.size() != 3) {
            throw std::invalid_argument("CIR requires 3 parameters: kappa, theta, sigma");
        }
        auto newParams = std::make_unique<CIRParameters>(params[0], params[1], params[2]);
        newParams->validate();
        cirParams_ = newParams.get();
        params_ = std::move(newParams);
    }

    // Vraisemblance exacte via densité de transition chi² non-centrale
    double logLikelihood(const std::vector<double>& observations, double dt) const override {
        if (observations.size() < 2) return 0.0;

        double kappa = cirParams_->getKappa();
        double theta = cirParams_->getTheta();
        double sigma = cirParams_->getSigma();

        double logLik = 0.0;
        double expKappaDt = std::exp(-kappa * dt);
        double c = sigma * sigma * (1.0 - expKappaDt) / (4.0 * kappa);
        double d = 4.0 * kappa * theta / (sigma * sigma);

        for (size_t i = 1; i < observations.size(); ++i) {
            double r_prev = std::max(observations[i - 1], 1e-10);
            double r_curr = std::max(observations[i], 1e-10);

            double lambda = 4.0 * kappa * r_prev * expKappaDt
                          / (sigma * sigma * (1.0 - expKappaDt));

            double x = r_curr / c;
            double v = (d - 2.0) / 2.0;
            double sqrtLambdaX = std::sqrt(lambda * x);

            double logDensity = -std::log(2.0)
                              - (x + lambda) / 2.0
                              + (v / 2.0) * std::log(x / lambda)
                              + logBesselI(v, sqrtLambdaX)
                              - std::log(c);

            if (std::isfinite(logDensity)) {
                logLik += logDensity;
            } else {
                // Approximation gaussienne en cas d'instabilité numérique
                double mean = theta + (r_prev - theta) * expKappaDt;
                double var = (sigma * sigma / (2.0 * kappa))
                           * (1.0 - std::exp(-2.0 * kappa * dt))
                           + (r_prev * sigma * sigma * expKappaDt / kappa)
                           * (1.0 - expKappaDt);
                double sd = std::sqrt(std::max(var, 1e-10));
                logLik += -0.5 * std::log(2.0 * M_PI * var)
                        - 0.5 * (r_curr - mean) * (r_curr - mean) / var;
            }
        }

        return logLik;
    }

    // Moyenne de la distribution stationnaire
    double getStationaryMean() const {
        return cirParams_->getTheta();
    }

    // Variance de la distribution stationnaire
    double getStationaryVariance() const {
        double kappa = cirParams_->getKappa();
        double theta = cirParams_->getTheta();
        double sigma = cirParams_->getSigma();
        return theta * sigma * sigma / (2.0 * kappa);
    }

    // Vérifie si la condition de Feller est satisfaite
    bool fellerConditionSatisfied() const {
        double kappa = cirParams_->getKappa();
        double theta = cirParams_->getTheta();
        double sigma = cirParams_->getSigma();
        return 2.0 * kappa * theta >= sigma * sigma;
    }

    std::shared_ptr<StochasticProcess> clone() const override {
        auto paramsClone = std::make_unique<CIRParameters>(*cirParams_);
        return std::make_shared<CoxIngersollRoss>(initialState_, std::move(paramsClone), 0);
    }
};

}
