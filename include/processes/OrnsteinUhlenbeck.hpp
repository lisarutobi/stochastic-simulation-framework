#pragma once

#include "core/StochasticProcess.hpp"
#include "core/ProcessParameters.hpp"
#include <cmath>
#include <limits>

namespace stochastic {

class OrnsteinUhlenbeck : public StochasticProcess {
private:
    OUParameters* ouParams_;
    
public:
    explicit OrnsteinUhlenbeck(
        double initialState = 0.0,
        double theta = 1.0,
        double mu = 0.0,
        double sigma = 0.1,
        unsigned long seed = 0
    ) : StochasticProcess(initialState, seed) {
        auto params = std::make_unique<OUParameters>(theta, mu, sigma);
        params->validate();
        ouParams_ = params.get();
        params_ = std::move(params);
    }
    
    OrnsteinUhlenbeck(
        double initialState,
        std::unique_ptr<OUParameters> params,
        unsigned long seed = 0
    ) : StochasticProcess(initialState, seed) {
        params->validate();
        ouParams_ = params.get();
        params_ = std::move(params);
    }
    
    double nextStep(double dt) override {
        double x = currentState_;
        double drift = ouParams_->getTheta() * (ouParams_->getMu() - x) * dt;
        double diffusion = ouParams_->getSigma() * std::sqrt(dt) * generateGaussian();
        x += drift + diffusion;
        currentState_ = x;
        return x;
    }
    
    double drift(double t, double x) const override {
        (void)t;
        return ouParams_->getTheta() * (ouParams_->getMu() - x);
    }
    
    double diffusion(double t, double x) const override {
        (void)t; (void)x;
        return ouParams_->getSigma();
    }
    
    std::string getName() const override {
        return "Ornstein-Uhlenbeck (OU)";
    }
    
    void validateParameters() const override {
        ouParams_->validate();
    }
    
    bool hasStationaryDistribution() const override {
        return true;
    }
    
    bool canBeNegative() const override {
        return true;
    }
    
    std::vector<double> simulatePathExact(const TimeGrid& timeGrid) override {
        reset();
        std::vector<double> path;
        path.reserve(timeGrid.size());
        path.push_back(currentState_);
        
        double prev_x = currentState_;
        for (size_t i = 0; i < timeGrid.getNumSteps(); ++i) {
            double dt = timeGrid.getIncrement(i);
            double expTheta = std::exp(-ouParams_->getTheta() * dt);
            double mean = ouParams_->getMu() + (prev_x - ouParams_->getMu()) * expTheta;
            double variance = (ouParams_->getSigma() * ouParams_->getSigma() / (2.0 * ouParams_->getTheta())) * 
                              (1.0 - expTheta * expTheta);
            double stddev = std::sqrt(variance);
            double x_next = mean + stddev * generateGaussian();
            currentState_ = x_next;
            path.push_back(x_next);
            prev_x = x_next;
        }
        return path;
    }
    
    std::vector<std::string> getParameterNames() const override {
        return {"theta", "mu", "sigma"};
    }
    
    std::vector<double> getParametersVector() const override {
        return {ouParams_->getTheta(), ouParams_->getMu(), ouParams_->getSigma()};
    }
    
    void setParametersVector(const std::vector<double>& params) override {
        if (params.size() != 3) {
            throw std::invalid_argument("OU requires 3 parameters: theta, mu, sigma");
        }
        auto newParams = std::make_unique<OUParameters>(params[0], params[1], params[2]);
        newParams->validate();
        ouParams_ = newParams.get();
        params_ = std::move(newParams);
    }
    
    double logLikelihood(const std::vector<double>& observations, double dt) const override {
        if (observations.size() < 2) return 0.0;
        
        double logLik = 0.0;
        double prev_x = observations[0];
        
        for (size_t i = 1; i < observations.size(); ++i) {
            double x = observations[i];
            double expTheta = std::exp(-ouParams_->getTheta() * dt);
            double mean = ouParams_->getMu() + (prev_x - ouParams_->getMu()) * expTheta;
            double variance = (ouParams_->getSigma() * ouParams_->getSigma() / (2.0 * ouParams_->getTheta())) * 
                              (1.0 - expTheta * expTheta);
            if (variance <= 0) return -std::numeric_limits<double>::infinity();
            double z = (x - mean) / std::sqrt(variance);
            logLik += -0.5 * std::log(2.0 * M_PI * variance) - 0.5 * z * z;
            prev_x = x;
        }
        
        return logLik;
    }
    
    std::shared_ptr<StochasticProcess> clone() const override {
        auto paramsClone = std::make_unique<OUParameters>(*ouParams_);
        return std::make_shared<OrnsteinUhlenbeck>(initialState_, std::move(paramsClone), 0);
    }
};

} 