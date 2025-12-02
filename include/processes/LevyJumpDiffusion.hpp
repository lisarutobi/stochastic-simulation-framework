#include "core/StochasticProcess.hpp"
#include "core/ProcessParameters.hpp"
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <memory>
#include <random>

namespace stochastic {

class LevyJumpDiffusion : public StochasticProcess {
private:
    LevyJumpParameters* levyParams_;
    std::poisson_distribution<int> poisson_;  // For number of jumps
    
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
        poisson_(lambda) {  // lambda is rate, but for dt we adjust
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
        
        // Diffusion part
        double drift = (mu - 0.5 * sigma * sigma - lambda * (std::exp(jumpMean + 0.5 * jumpStd * jumpStd) - 1.0)) * dt;
        double diffusion = sigma * std::sqrt(dt) * generateGaussian();
        S *= std::exp(drift + diffusion);
        
        // Jump part
        int numJumps = poisson_(generator_) * dt;  // Approximate for small dt
        for (int j = 0; j < numJumps; ++j) {
            double jump = jumpMean + jumpStd * generateGaussian();
            S *= (1.0 + jump);
        }
        
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
        return "Levy Jump Diffusion";
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
    // For Merton model, exact is possible but similar to approx for small dt
        return simulatePath(timeGrid, false);  // false = pas d'exact
    }


    std::vector<std::string> getParameterNames() const override {
        return {"mu", "sigma", "lambda", "jumpMean", "jumpStd"};
    }
    
    std::vector<double> getParametersVector() const override {
        return {levyParams_->getMu(), levyParams_->getSigma(), levyParams_->getLambda(),
                levyParams_->getJumpMean(), levyParams_->getJumpStd()};
    }
    
    void setParametersVector(const std::vector<double>& params) override {
        if (params.size() != 5) {
            throw std::invalid_argument("Levy requires 5 parameters");
        }
        auto newParams = std::make_unique<LevyJumpParameters>(params[0], params[1], params[2], params[3], params[4]);
        newParams->validate();
        levyParams_ = newParams.get();
        params_ = std::move(newParams);
    }
    
    double logLikelihood(const std::vector<double>& observations, double dt) const override {
        // Complex for jumps; placeholder
        return 0.0;
    }
    
    std::shared_ptr<StochasticProcess> clone() const override {
        auto paramsClone = std::make_unique<LevyJumpParameters>(*levyParams_);
        return std::make_shared<LevyJumpDiffusion>(initialState_, std::move(paramsClone), 0);
    }
};

}