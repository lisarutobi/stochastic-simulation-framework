#include "core/StochasticProcess.hpp"
#include "core/ProcessParameters.hpp"
#include <cmath>
#include <limits>

namespace stochastic {

class GeometricBrownianMotion : public StochasticProcess {
private:
    GBMParameters* gbmParams_;
    
public:
    explicit GeometricBrownianMotion(
        double initialState = 100.0,
        double mu = 0.05,
        double sigma = 0.2,
        unsigned long seed = 0
    ) : StochasticProcess(initialState, seed) {
        auto params = std::make_unique<GBMParameters>(mu, sigma);
        params->validate();
        gbmParams_ = params.get();
        params_ = std::move(params);
    }
    
    GeometricBrownianMotion(
        double initialState,
        std::unique_ptr<GBMParameters> params,
        unsigned long seed = 0
    ) : StochasticProcess(initialState, seed) {
        params->validate();
        gbmParams_ = params.get();
        params_ = std::move(params);
    }
    
    double nextStep(double dt) override {
        double mu = gbmParams_->getMu();
        double sigma = gbmParams_->getSigma();
        double Z = generateGaussian();
        double newState = currentState_ * (1.0 + mu * dt + sigma * std::sqrt(dt) * Z);
        return newState;
    }
    
    double drift(double t, double x) const override {
        (void)t;
        return gbmParams_->getMu() * x;
    }
    
    double diffusion(double t, double x) const override {
        (void)t;
        return gbmParams_->getSigma() * x;
    }
    
    std::string getName() const override {
        return "Geometric Brownian Motion (GBM)";
    }
    
    void validateParameters() const override {
        gbmParams_->validate();
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
    
    double exactSolution(double t, double W_t) const {
        double mu = gbmParams_->getMu();
        double sigma = gbmParams_->getSigma();
        return initialState_ * std::exp((mu - 0.5 * sigma * sigma) * t + sigma * W_t);
    }
    
    std::vector<double> simulatePathExact(const TimeGrid& timeGrid) override {
        reset();
        std::vector<double> path;
        path.reserve(timeGrid.size());
        path.push_back(currentState_);
        
        double W_t = 0.0;
        for (size_t i = 0; i < timeGrid.getNumSteps(); ++i) {
            double dt = timeGrid.getIncrement(i);
            double t = timeGrid[i+1];
            double dW = std::sqrt(dt) * generateGaussian();
            W_t += dW;
            currentState_ = exactSolution(t, W_t);
            path.push_back(currentState_);
        }
        return path;
    }
    
    double theoreticalMean(double t) const {
        return initialState_ * std::exp(gbmParams_->getMu() * t);
    }
    
    double theoreticalVariance(double t) const {
        double mu = gbmParams_->getMu();
        double sigma = gbmParams_->getSigma();
        double S0 = initialState_;
        return S0 * S0 * std::exp(2.0 * mu * t) * (std::exp(sigma * sigma * t) - 1.0);
    }
    
    double theoreticalStdDev(double t) const {
        return std::sqrt(theoreticalVariance(t));
    }
    
    // Méthodes pour MCMC
    std::vector<std::string> getParameterNames() const override {
        return {"mu", "sigma"};
    }
    
    std::vector<double> getParametersVector() const override {
        return {gbmParams_->getMu(), gbmParams_->getSigma()};
    }
    
    void setParametersVector(const std::vector<double>& params) override {
        if (params.size() != 2) {
            throw std::invalid_argument("GBM requires 2 parameters: mu, sigma");
        }
        gbmParams_->setMu(params[0]);
        gbmParams_->setSigma(params[1]);
    }
    
    double logLikelihood(const std::vector<double>& observations, double dt) const override {
        if (observations.size() < 2) return 0.0;
        
        double mu = gbmParams_->getMu();
        double sigma = gbmParams_->getSigma();
        
        if (sigma <= 0) {
            return -std::numeric_limits<double>::infinity();
        }
        
        double logLik = 0.0;
        
        // Check if observations are log-returns or prices
        bool isReturns = true;
        for (size_t i = 0; i < std::min(size_t(10), observations.size()); ++i) {
            if (observations[i] > 10.0) {  // Heuristic: returns are typically small
                isReturns = false;
                break;
            }
        }
        
        if (isReturns) {
            // Direct log-return likelihood (much more stable!)
            double expectedReturn = (mu - 0.5 * sigma * sigma) * dt;
            double variance = sigma * sigma * dt;
            
            if (variance <= 0) {
                return -std::numeric_limits<double>::infinity();
            }
            
            for (size_t i = 0; i < observations.size(); ++i) {
                double logReturn = observations[i];
                double z = (logReturn - expectedReturn) / std::sqrt(variance);
                logLik += -0.5 * std::log(2.0 * M_PI * variance) - 0.5 * z * z;
            }
        } else {
            // Price-based likelihood (original method)
            for (size_t i = 0; i < observations.size() - 1; ++i) {
                double S_t = observations[i];
                double S_tplus1 = observations[i + 1];
                
                if (S_t <= 0 || S_tplus1 <= 0) {
                    return -std::numeric_limits<double>::infinity();
                }
                
                double logReturn = std::log(S_tplus1 / S_t);
                double mean = (mu - 0.5 * sigma * sigma) * dt;
                double variance = sigma * sigma * dt;
                
                if (variance <= 0) {
                    return -std::numeric_limits<double>::infinity();
                }
                
                double z = (logReturn - mean) / std::sqrt(variance);
                logLik += -0.5 * std::log(2.0 * M_PI * variance) - 0.5 * z * z;
            }
        }
        
        return logLik;
    }
    
    std::shared_ptr<StochasticProcess> clone() const override {
        return std::make_shared<GeometricBrownianMotion>(
            initialState_, gbmParams_->getMu(), gbmParams_->getSigma(), 0
        );
    }
    
    // Accesseurs
    double getMu() const { return gbmParams_->getMu(); }
    double getSigma() const { return gbmParams_->getSigma(); }
    
    void setMu(double mu) { 
        gbmParams_->setMu(mu); 
        validateParameters();
    }
    
    void setSigma(double sigma) { 
        gbmParams_->setSigma(sigma);
        validateParameters();
    }
    
    void setGBMParameters(double mu, double sigma) {
        auto newParams = std::make_unique<GBMParameters>(mu, sigma);
        newParams->validate();
        gbmParams_ = newParams.get();
        params_ = std::move(newParams);
    }
};

} 