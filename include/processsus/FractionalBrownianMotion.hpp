#pragma once
#include "../core/StochasticProcess.hpp"
#include <vector>
#include <cmath>

namespace stochastic {

class FractionalBrownianMotion : public StochasticProcess {
private:
    double H_;
    double sigma_;

public:
    FractionalBrownianMotion(double init = 0.0, double H = 0.5, double sigma = 1.0, unsigned long seed = 0)
        : StochasticProcess(init, seed), H_(H), sigma_(sigma) {
        if (H <= 0.0 || H >= 1.0 || sigma <= 0.0)
            throw std::invalid_argument("Invalid fBM parameters");
    }

    std::string getName() const override {
        return "Fractional Brownian Motion (H=" + std::to_string(H_) + ")";
    }

    double drift(double, double) const override { return 0.0; }
    double diffusion(double, double) const override { return sigma_; }
    void validateParameters() const override {}

    double nextStep(double) override {
        throw std::runtime_error("fBM is not Markovian — use simulatePath() with full grid");
    }

    //version simplifiée : on simule une fBM "approximative" en utilisant des incréments gaussiens
    std::vector<double> simulatePathExact(const TimeGrid& grid) override {
        size_t N = grid.getNumSteps();
        std::vector<double> path(N + 1);
        path[0] = initialState_;

        double dt = grid.getFinalTime() / N;
        double scale = sigma_ * std::pow(dt, H_);  // ← la clé magique du fBM

        for (size_t i = 1; i <= N; ++i) {
            path[i] = path[i-1] + scale * generateGaussian();
        }

        currentState_ = path.back();
        return path;
    }

    // On force l'exact
    std::vector<double> simulatePath(const TimeGrid& grid, bool = false) override {
        return simulatePathExact(grid);
    }

    std::vector<double> simulatePath(double T, size_t nSteps, bool = false) override {
        TimeGrid g(0.0, T, nSteps);
        return simulatePathExact(g);
    }

    std::shared_ptr<StochasticProcess> clone() const override {
        auto clone = std::make_shared<FractionalBrownianMotion>(initialState_, H_, sigma_);
        clone->setSeed(std::random_device{}());
        return clone;
    }

    std::vector<std::string> getParameterNames() const override { return {"H", "sigma"}; }
    std::vector<double> getParametersVector() const override { return {H_, sigma_}; }
    void setParametersVector(const std::vector<double>& p) override { H_ = p[0]; sigma_ = p[1]; }
    double logLikelihood(const std::vector<double>&, double) const override { return 0.0; }
};

} 