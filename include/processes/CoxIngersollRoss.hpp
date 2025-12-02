#include "core/StochasticProcess.hpp"
#include "core/ProcessParameters.hpp"
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <algorithm>  // for std::min

namespace stochastic {

class CoxIngersollRoss : public StochasticProcess {
private:
    CIRParameters* cirParams_;
    
public:
    explicit CoxIngersollRoss(
        double initialRate = 0.03,
        double kappa = 0.5,
        double theta = 0.03,
        double sigma = 0.1,
        unsigned long seed = 0
    ) : StochasticProcess(initialRate, seed) {
        auto params = std::make_unique<CIRParameters>(kappa, theta, sigma);
        params->validate();
        cirParams_ = params.get();
        params_ = std::move(params);
    }
    
    CoxIngersollRoss(
        double initialRate,
        std::unique_ptr<CIRParameters> params,
        unsigned long seed = 0
    ) : StochasticProcess(initialRate, seed) {
        params->validate();
        cirParams_ = params.get();
        params_ = std::move(params);
    }
    
    // Next step using Euler scheme (may go negative; use truncation)
    double nextStep(double dt) override {
        double r = currentState_;
        double kappa = cirParams_->getKappa();
        double theta = cirParams_->getTheta();
        double sigma = cirParams_->getSigma();
        
        double dr = kappa * (theta - r) * dt + sigma * std::sqrt(std::max(r, 0.0)) * std::sqrt(dt) * generateGaussian();
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
    
    // Exact simulation for CIR (non-central chi-squared)
    std::vector<double> simulatePathExact(const TimeGrid& timeGrid) override {
        reset();
        std::vector<double> path;
        path.reserve(timeGrid.size());
        path.push_back(currentState_);
        
        double prev_r = currentState_;
        for (size_t i = 0; i < timeGrid.getNumSteps(); ++i) {
            double dt = timeGrid.getIncrement(i);
            double kappa = cirParams_->getKappa();
            double theta = cirParams_->getTheta();
            double sigma = cirParams_->getSigma();
            
            double df = 4.0 * kappa * theta / (sigma * sigma);  // Degrees of freedom
            double lambda = 4.0 * kappa * prev_r * std::exp(-kappa * dt) / (sigma * sigma * (1.0 - std::exp(-kappa * dt)));
            
            // Generate non-central chi-squared
            // Approximation or use gamma/poisson; here simple approx using normal for simplicity
            // TODO: Implement proper non-central chi2
            double mean = theta + (prev_r - theta) * std::exp(-kappa * dt);
            double var = (sigma * sigma / (2.0 * kappa)) * (1.0 - std::exp(-2.0 * kappa * dt)) + 
                         (prev_r * sigma * sigma * std::exp(-kappa * dt) / kappa) * (1.0 - std::exp(-kappa * dt));
            double stddev = std::sqrt(var);
            double r_next = mean + stddev * generateGaussian();
            
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
    
    double logLikelihood(const std::vector<double>& observations, double dt) const override {
        // TODO: Implement exact transition density for CIR (non-central chi2)
        return 0.0;  // Placeholder
    }
    
    std::shared_ptr<StochasticProcess> clone() const override {
        auto paramsClone = std::make_unique<CIRParameters>(*cirParams_);
        return std::make_shared<CoxIngersollRoss>(initialState_, std::move(paramsClone), 0);
    }
};

}