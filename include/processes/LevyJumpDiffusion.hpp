#pragma once

#include "core/StochasticProcess.hpp"
#include "core/ProcessParameters.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

namespace stochastic {

class LevyJumpDiffusion : public StochasticProcess {
private:
    LevyJumpParameters* levyParams_;
    std::poisson_distribution<int> poisson_;

    // Densité log-normale
    static double normalLogDensity(double x, double mean, double variance) {
        if (variance <= 0) return -std::numeric_limits<double>::infinity();
        return -0.5 * std::log(2.0 * M_PI * variance)
             - 0.5 * (x - mean) * (x - mean) / variance;
    }

    // Vraisemblance pour une observation (somme sur le nombre de sauts possibles)
    double logLikelihoodSingleObs(double logReturn, double dt) const {
        double mu = levyParams_->getMu();
        double sigma = levyParams_->getSigma();
        double lambda = levyParams_->getLambda();
        double jumpMean = levyParams_->getJumpMean();
        double jumpStd = levyParams_->getJumpStd();

        // Terme de compensation k = E[e^Y - 1]
        double k = std::exp(jumpMean + 0.5 * jumpStd * jumpStd) - 1.0;

        // Drift et variance de base (sans sauts)
        double baseDrift = (mu - 0.5 * sigma * sigma - lambda * k) * dt;
        double baseVar = sigma * sigma * dt;

        // Somme sur les nombres de sauts possibles (troncature raisonnable)
        std::vector<double> logTerms;
        double lambdaDt = lambda * dt;
        double logLambdaDt = std::log(lambdaDt + 1e-300);

        int maxJumps = static_cast<int>(lambdaDt + 6.0 * std::sqrt(lambdaDt) + 5);
        maxJumps = std::max(maxJumps, 10);

        double logPoisson = -lambdaDt;

        for (int n = 0; n <= maxJumps; ++n) {
            if (n > 0) {
                logPoisson += logLambdaDt - std::log(static_cast<double>(n));
            }

            double condMean = baseDrift + n * jumpMean;
            double condVar = baseVar + n * jumpStd * jumpStd;

            double logNormal = normalLogDensity(logReturn, condMean, condVar);
            logTerms.push_back(logPoisson + logNormal);
        }

        // Log-sum-exp pour stabilité numérique
        double maxLog = *std::max_element(logTerms.begin(), logTerms.end());
        double sumExp = 0.0;
        for (double logT : logTerms) {
            sumExp += std::exp(logT - maxLog);
        }

        return maxLog + std::log(sumExp);
    }

public:
    explicit LevyJumpDiffusion(
        double initialState = 100.0,
        double mu = 0.05,
        double sigma = 0.2,
        double lambda = 0.1,
        double jumpMean = -0.05,
        double jumpStd = 0.1,
        unsigned long seed = 0
    ) : StochasticProcess(initialState, seed),
        poisson_(lambda) {
        auto params = std::make_unique<LevyJumpParameters>(mu, sigma, lambda, jumpMean, jumpStd);
        params->validate();
        levyParams_ = params.get();
        params_ = std::move(params);
    }

    LevyJumpDiffusion(
        double initialState,
        std::unique_ptr<LevyJumpParameters> params,
        unsigned long seed = 0
    ) : StochasticProcess(initialState, seed),
        poisson_(params->getLambda()) {
        params->validate();
        levyParams_ = params.get();
        params_ = std::move(params);
    }

    double nextStep(double dt) override {
        double S = currentState_;
        double mu = levyParams_->getMu();
        double sigma = levyParams_->getSigma();
        double lambda = levyParams_->getLambda();
        double jumpMean = levyParams_->getJumpMean();
        double jumpStd = levyParams_->getJumpStd();

        // Compensation k = E[e^Y - 1]
        double k = std::exp(jumpMean + 0.5 * jumpStd * jumpStd) - 1.0;

        // Partie diffusion (compensée pour les sauts)
        double drift = (mu - 0.5 * sigma * sigma - lambda * k) * dt;
        double diffusion = sigma * std::sqrt(dt) * generateGaussian();

        // Partie sauts : N ~ Poisson(lambda * dt)
        std::poisson_distribution<int> poissonDt(lambda * dt);
        int numJumps = poissonDt(generator_);

        double jumpSum = 0.0;
        for (int j = 0; j < numJumps; ++j) {
            jumpSum += jumpMean + jumpStd * generateGaussian();
        }

        S *= std::exp(drift + diffusion + jumpSum);

        currentState_ = S;
        enforceConstraints();
        return S;
    }

    double drift(double t, double x) const override {
        (void)t;
        return levyParams_->getMu() * x;
    }

    double diffusion(double t, double x) const override {
        (void)t;
        return levyParams_->getSigma() * x;
    }

    std::string getName() const override {
        return "Levy Jump Diffusion (Merton)";
    }

    void validateParameters() const override {
        levyParams_->validate();
    }

    bool hasStationaryDistribution() const override {
        return false;
    }

    bool canBeNegative() const override {
        return false;
    }

    void enforceConstraints() override {
        if (currentState_ <= 0.0) {
            currentState_ = 1e-10;
        }
    }

    std::vector<double> simulatePathExact(const TimeGrid& timeGrid) override {
        reset();
        std::vector<double> path;
        path.reserve(timeGrid.size());
        path.push_back(currentState_);

        double mu = levyParams_->getMu();
        double sigma = levyParams_->getSigma();
        double lambda = levyParams_->getLambda();
        double jumpMean = levyParams_->getJumpMean();
        double jumpStd = levyParams_->getJumpStd();

        double k = std::exp(jumpMean + 0.5 * jumpStd * jumpStd) - 1.0;

        for (size_t i = 0; i < timeGrid.getNumSteps(); ++i) {
            double dt = timeGrid.getIncrement(i);

            double drift = (mu - 0.5 * sigma * sigma - lambda * k) * dt;
            double diffusion = sigma * std::sqrt(dt) * generateGaussian();

            std::poisson_distribution<int> poissonDt(lambda * dt);
            int numJumps = poissonDt(generator_);

            double jumpSum = 0.0;
            for (int j = 0; j < numJumps; ++j) {
                jumpSum += jumpMean + jumpStd * generateGaussian();
            }

            currentState_ *= std::exp(drift + diffusion + jumpSum);
            enforceConstraints();
            path.push_back(currentState_);
        }

        return path;
    }

    std::vector<std::string> getParameterNames() const override {
        return {"mu", "sigma", "lambda", "jumpMean", "jumpStd"};
    }

    std::vector<double> getParametersVector() const override {
        return {
            levyParams_->getMu(),
            levyParams_->getSigma(),
            levyParams_->getLambda(),
            levyParams_->getJumpMean(),
            levyParams_->getJumpStd()
        };
    }

    void setParametersVector(const std::vector<double>& params) override {
        if (params.size() != 5) {
            throw std::invalid_argument("Levy requires 5 parameters: mu, sigma, lambda, jumpMean, jumpStd");
        }
        auto newParams = std::make_unique<LevyJumpParameters>(
            params[0], params[1], params[2], params[3], params[4]
        );
        newParams->validate();
        levyParams_ = newParams.get();
        params_ = std::move(newParams);

        poisson_ = std::poisson_distribution<int>(levyParams_->getLambda());
    }

    // Vraisemblance via densité mélange (somme sur nombre de sauts)
    double logLikelihood(const std::vector<double>& observations, double dt) const override {
        if (observations.size() < 2) return 0.0;

        double logLik = 0.0;

        for (size_t i = 1; i < observations.size(); ++i) {
            double logReturn = std::log(observations[i] / observations[i - 1]);
            logLik += logLikelihoodSingleObs(logReturn, dt);
        }

        return logLik;
    }

    // Vraisemblance directe à partir des log-rendements
    double logLikelihoodFromReturns(const std::vector<double>& logReturns, double dt) const {
        double logLik = 0.0;
        for (double r : logReturns) {
            logLik += logLikelihoodSingleObs(r, dt);
        }
        return logLik;
    }

    // Taille moyenne attendue des sauts
    double getExpectedJumpSize() const {
        double jumpMean = levyParams_->getJumpMean();
        double jumpStd = levyParams_->getJumpStd();
        return std::exp(jumpMean + 0.5 * jumpStd * jumpStd) - 1.0;
    }

    // Intensité des sauts (nombre moyen par an)
    double getJumpIntensity() const {
        return levyParams_->getLambda();
    }

    // Variance totale incluant la contribution des sauts
    double getTotalVariance(double dt) const {
        double sigma = levyParams_->getSigma();
        double lambda = levyParams_->getLambda();
        double jumpMean = levyParams_->getJumpMean();
        double jumpStd = levyParams_->getJumpStd();

        double jumpSecondMoment = jumpMean * jumpMean + jumpStd * jumpStd;
        return sigma * sigma * dt + lambda * dt * jumpSecondMoment;
    }

    std::shared_ptr<StochasticProcess> clone() const override {
        auto paramsClone = std::make_unique<LevyJumpParameters>(*levyParams_);
        return std::make_shared<LevyJumpDiffusion>(initialState_, std::move(paramsClone), 0);
    }
};

}
