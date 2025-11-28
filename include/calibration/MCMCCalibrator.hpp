#pragma once
#include "../core/StochasticProcess.hpp"
#include "../data/TimeSeriesData.hpp"
#include <vector>
#include <map>
#include <string>
#include <random>
#include <cmath>
#include <memory>
#include <eigen3/Eigen/Dense>

class MCMCCalibrator {
public:
    struct Prior {
        enum class Type { UNIFORM, NORMAL, LOGNORMAL, GAMMA, INVERSE_GAMMA };
        Type type;
        std::vector<double> parameters;

        double logDensity(double x) const;
        bool inSupport(double x) const;
    };

    struct MCMCConfig {
        size_t nIterations = 50000;
        size_t burnIn = 10000;
        size_t thinning = 10;
        bool adaptiveProposal = true;
        double initialStepSize = 0.1;
        double targetAcceptanceRate = 0.234;
        size_t adaptationWindow = 100;
        std::map<std::string, Prior> priors;
    };

    struct CalibrationResult {
        std::map<std::string, double> meanParams;
        std::map<std::string, double> medianParams;
        std::map<std::string, std::pair<double, double>> credibleIntervals95;
        double gelmanRubinStatistic;
        double effectiveSampleSize;
        double acceptanceRate;
        double logLikelihood;
        double AIC;
        double BIC;
        double DIC;
    };

    MCMCCalibrator(std::shared_ptr<stochastic::StochasticProcess> process,
                   const TimeSeriesData& data,
                   const MCMCConfig& config);

    CalibrationResult calibrateFromMarketData();

private:
    std::shared_ptr<stochastic::StochasticProcess> process_;
    TimeSeriesData observedData_;
    MCMCConfig config_;

    std::vector<double> currentParams_;
    std::vector<std::vector<double>> chain_;
    size_t nAccepted_;
    Eigen::MatrixXd proposalCovariance_;
    std::mt19937 rng_;
    std::normal_distribution<double> normalDist_;

    void initializeParameters();
    double sampleFromPrior(const Prior& prior);
    void runMetropolisHastings();
    std::vector<double> proposeParameters(const std::vector<double>& current);
    void adaptProposal(size_t iter);
    double computeLogLikelihood(const std::vector<double>& params) const;
    double computeLogPrior(const std::vector<double>& params) const;
    double computeLogPosterior(const std::vector<double>& params) const;
    bool inPriorSupport(const std::vector<double>& params) const;
    CalibrationResult processResults();
    double computeMean(const std::vector<std::vector<double>>& samples, size_t idx) const;
    double computeMedian(const std::vector<std::vector<double>>& samples, size_t idx) const;
    std::pair<double,double> computeCredibleInterval(const std::vector<std::vector<double>>& samples, size_t idx, double level) const;
};
