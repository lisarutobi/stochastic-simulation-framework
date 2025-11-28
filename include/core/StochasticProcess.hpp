// include/core/StochasticProcess.hpp
#ifndef STOCHASTIC_PROCESS_HPP
#define STOCHASTIC_PROCESS_HPP

#include "ProcessParameters.hpp"
#include "TimeGrid.hpp"          // ← ON UTILISE LA VRAIE CLASSE ICI
#include <vector>
#include <random>
#include <memory>
#include <string>

namespace stochastic {

class StochasticProcess {
protected:
    double currentState_;
    double initialState_;
    std::unique_ptr<ProcessParameters> params_;

    std::mt19937 generator_;
    std::normal_distribution<double> stdNormal_;

public:
    explicit StochasticProcess(double initialState = 0.0, unsigned long seed = 0)
        : currentState_(initialState),
          initialState_(initialState),
          stdNormal_(0.0, 1.0) {
        if (seed == 0) {
            std::random_device rd;
            generator_.seed(rd());
            } 
        else {
            generator_.seed(seed);
            }
    }

    virtual ~StochasticProcess() = default;

    // No copy, only move
    StochasticProcess(const StochasticProcess&) = delete;
    StochasticProcess& operator=(const StochasticProcess&) = delete;
    StochasticProcess(StochasticProcess&&) = default;
    StochasticProcess& operator=(StochasticProcess&&) = default;

    // ====================================================================
    // Pure virtual methods
    // ====================================================================
    virtual double nextStep(double dt) = 0;
    virtual double drift(double t, double x) const = 0;
    virtual double diffusion(double t, double x) const = 0;
    virtual std::string getName() const = 0;
    virtual void validateParameters() const = 0;

    virtual std::vector<std::string> getParameterNames() const = 0;
    virtual std::vector<double> getParametersVector() const = 0;
    virtual void setParametersVector(const std::vector<double>& params) = 0;
    virtual double logLikelihood(const std::vector<double>& obs, double dt) const = 0;

    virtual std::shared_ptr<StochasticProcess> clone() const = 0;

    // ====================================================================
    // Default virtual methods
    // ====================================================================
    virtual bool hasStationaryDistribution() const { return false; }
    virtual bool canBeNegative() const { return true; }
    virtual void enforceConstraints() {}

    virtual std::vector<double> simulatePathExact(const TimeGrid& grid) {
        return simulatePath(grid, false);  // fallback Euler si pas d’exact
    }

    // ====================================================================
    // MÉTHODE PRINCIPALE : simulation avec TimeGrid
    // ====================================================================
    virtual std::vector<double> simulatePath(const TimeGrid& grid, bool useExact = false) {
        if (useExact) {
            return simulatePathExact(grid);
        }

        reset();
        std::vector<double> path;
        path.reserve(grid.size());
        path.push_back(currentState_);

        for (size_t i = 0; i < grid.getNumSteps(); ++i) {
            double dt = grid.getIncrement(i);
            currentState_ = nextStep(dt);
            enforceConstraints();
            path.push_back(currentState_);
        }
        return path;
    }

    // Surcharges compatibilité ascendante
    virtual std::vector<double> simulatePath(double T, size_t numSteps, bool useExact = false) {
        TimeGrid grid(0.0, T, numSteps);
        return simulatePath(grid, useExact);
    }

    virtual std::vector<double> simulatePath(double T, size_t numSteps) {
        return simulatePath(T, numSteps, false);
    }

    // Plusieurs trajectoires
    std::vector<std::vector<double>> simulatePaths(double T, size_t numSteps, size_t nPaths, bool useExact = false) {
        std::vector<std::vector<double>> paths;
        paths.reserve(nPaths);
        for (size_t i = 0; i < nPaths; ++i) {
            paths.push_back(simulatePath(T, numSteps, useExact));
            reset();
        }
        return paths;
    }

    // ====================================================================
    // Utilitaires
    // ====================================================================
    void reset() { currentState_ = initialState_; }
    double getCurrentState() const { return currentState_; }
    void setCurrentState(double s) { currentState_ = s; }
    double getInitialState() const { return initialState_; }
    void setInitialState(double s) { initialState_ = s; reset(); }

    void setSeed(unsigned long seed) {
        if (seed == 0) {
            std::random_device rd;
            generator_.seed(rd());
        } else {
            generator_.seed(seed);
        }
    }

    double generateGaussian() { return stdNormal_(generator_); }

    virtual std::string toString() const {
        return "Process: " + getName();
    }
};

} // namespace stochastic

#endif // STOCHASTIC_PROCESS_HPP