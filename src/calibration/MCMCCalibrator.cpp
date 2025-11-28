// src/calibration/MCMCCalibrator.cpp
#include "calibration/MCMCCalibrator.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

double MCMCCalibrator::Prior::logDensity(double x) const {
    switch(type) {
        case Type::UNIFORM: {
            double a = parameters[0], b = parameters[1];
            return (x >= a && x <= b) ? -std::log(b - a) : -INFINITY;
        }
        case Type::NORMAL: {
            double mu = parameters[0], sigma = parameters[1];
            double z = (x - mu) / sigma;
            return -0.5 * std::log(2 * M_PI * sigma * sigma) - 0.5 * z * z;
        }
        case Type::LOGNORMAL: {
            if (x <= 0) return -INFINITY;
            double mu = parameters[0], sigma = parameters[1];
            double z = (std::log(x) - mu) / sigma;
            return -std::log(x) - 0.5 * std::log(2 * M_PI * sigma * sigma) - 0.5 * z * z;
        }
        default: return 0.0;
    }
}

bool MCMCCalibrator::Prior::inSupport(double x) const {
    switch(type) {
        case Type::UNIFORM: return x >= parameters[0] && x <= parameters[1];
        case Type::NORMAL: return true;
        case Type::LOGNORMAL: return x > 0;
        default: return true;
    }
}

MCMCCalibrator::MCMCCalibrator(std::shared_ptr<stochastic::StochasticProcess> process,
                               const TimeSeriesData& data,
                               const MCMCConfig& config)
    : process_(process), observedData_(data), config_(config),
      rng_(std::random_device{}()), normalDist_(0.0,1.0), nAccepted_(0)
{
    size_t nParams = process_->getParameterNames().size();
    proposalCovariance_ = Eigen::MatrixXd::Identity(nParams, nParams) * config_.initialStepSize*config_.initialStepSize;
}

MCMCCalibrator::CalibrationResult MCMCCalibrator::calibrateFromMarketData() {
    initializeParameters();
    runMetropolisHastings();
    return processResults();
}

void MCMCCalibrator::initializeParameters() {
    auto names = process_->getParameterNames();
    currentParams_.resize(names.size());
    for(size_t i=0;i<names.size();++i){
        if(config_.priors.count(names[i])) currentParams_[i] = sampleFromPrior(config_.priors.at(names[i]));
        else currentParams_[i] = 0.1;
    }
    process_->setParametersVector(currentParams_);
}

double MCMCCalibrator::sampleFromPrior(const Prior& prior) {
    switch(prior.type){
        case Prior::Type::UNIFORM:
            return prior.parameters[0] + (prior.parameters[1]-prior.parameters[0])*std::uniform_real_distribution<>(0,1)(rng_);
        case Prior::Type::NORMAL:
            return prior.parameters[0] + prior.parameters[1]*normalDist_(rng_);
        case Prior::Type::LOGNORMAL:
            return exp(prior.parameters[0] + prior.parameters[1]*normalDist_(rng_));
        default: return 0.1;
    }
}

void MCMCCalibrator::runMetropolisHastings() {
    chain_.clear();
    chain_.reserve(config_.nIterations);
    double currentLogPosterior = computeLogPosterior(currentParams_);

    for(size_t iter=0; iter<config_.nIterations; ++iter){
        auto proposed = proposeParameters(currentParams_);
        if(!inPriorSupport(proposed)){
            chain_.push_back(currentParams_);
            continue;
        }
        double logPosteriorProposed = computeLogPosterior(proposed);
        double logAlpha = logPosteriorProposed - currentLogPosterior;
        bool accept = false;
        if(logAlpha >= 0 || log(std::uniform_real_distribution<>(0,1)(rng_)) < logAlpha){
            currentParams_ = proposed;
            currentLogPosterior = logPosteriorProposed;
            accept = true;
            ++nAccepted_;
        }
        chain_.push_back(currentParams_);
        if(config_.adaptiveProposal && iter>0 && iter<config_.burnIn && iter%config_.adaptationWindow==0){
            adaptProposal(iter);
        }
    }
}

std::vector<double> MCMCCalibrator::proposeParameters(const std::vector<double>& current){
    size_t n = current.size();
    std::vector<double> proposal(n);
    Eigen::VectorXd currentVec(n), z(n);
    for(size_t i=0;i<n;++i) currentVec(i) = current[i];
    for(size_t i=0;i<n;++i) z(i) = normalDist_(rng_);
    Eigen::LLT<Eigen::MatrixXd> llt(proposalCovariance_);
    Eigen::VectorXd proposalVec = currentVec + llt.matrixL()*z;
    for(size_t i=0;i<n;++i) proposal[i] = proposalVec(i);
    return proposal;
}

void MCMCCalibrator::adaptProposal(size_t iter){
    size_t startIdx = (iter>1000)?iter-1000:0;
    size_t n = iter - startIdx;
    size_t d = currentParams_.size();
    if(n<2) return;
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(d);
    for(size_t i=startIdx;i<iter;++i) for(size_t j=0;j<d;++j) mean(j)+=chain_[i][j];
    mean /= n;
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(d,d);
    for(size_t i=startIdx;i<iter;++i){
        Eigen::VectorXd diff(d);
        for(size_t j=0;j<d;++j) diff(j)=chain_[i][j]-mean(j);
        cov+=diff*diff.transpose();
    }
    cov/=(n-1);
    double scale = 2.38*2.38/d;
    proposalCovariance_ = scale*cov + 1e-6*Eigen::MatrixXd::Identity(d,d);
}

double MCMCCalibrator::computeLogLikelihood(const std::vector<double>& params) const {
    auto copy = process_->clone();
    try {
        copy->setParametersVector(params);
        // On force la validité
        copy->validateParameters();
        double ll = copy->logLikelihood(observedData_.path, observedData_.dt);
        return std::isfinite(ll) ? ll : -INFINITY;
    } catch (...) {
        return -INFINITY;
    }
}

double MCMCCalibrator::computeLogPrior(const std::vector<double>& params) const {
    double logP=0.0;
    auto names = process_->getParameterNames();
    for(size_t i=0;i<params.size();++i){
        if(config_.priors.count(names[i])) logP+=config_.priors.at(names[i]).logDensity(params[i]);
    }
    return logP;
}

double MCMCCalibrator::computeLogPosterior(const std::vector<double>& params) const{
    return computeLogLikelihood(params)+computeLogPrior(params);
}

bool MCMCCalibrator::inPriorSupport(const std::vector<double>& params) const{
    auto names = process_->getParameterNames();
    for(size_t i=0;i<params.size();++i){
        if(config_.priors.count(names[i]) && !config_.priors.at(names[i]).inSupport(params[i]))
            return false;
    }
    return true;
}

MCMCCalibrator::CalibrationResult MCMCCalibrator::processResults(){
    CalibrationResult res;
    std::vector<std::vector<double>> samples;
    for(size_t i=config_.burnIn;i<chain_.size();i+=config_.thinning) samples.push_back(chain_[i]);
    auto names = process_->getParameterNames();
    for(size_t i=0;i<names.size();++i){
        res.meanParams[names[i]] = computeMean(samples,i);
        res.medianParams[names[i]] = computeMedian(samples,i);
        res.credibleIntervals95[names[i]] = computeCredibleInterval(samples,i,0.95);
    }
    res.acceptanceRate = static_cast<double>(nAccepted_)/config_.nIterations;
    res.effectiveSampleSize = samples.size();
    res.gelmanRubinStatistic = 1.0;
    res.logLikelihood = computeLogLikelihood(currentParams_);
    size_t nParams = names.size();
    size_t nObs = observedData_.path.size();
    res.AIC = -2*res.logLikelihood + 2*nParams;
    res.BIC = -2*res.logLikelihood + nParams*log(nObs);
    res.DIC = res.AIC;
    return res;
}

double MCMCCalibrator::computeMean(const std::vector<std::vector<double>>& samples, size_t idx) const {
    double sum=0.0;
    for(const auto& s:samples) sum+=s[idx];
    return sum/samples.size();
}

double MCMCCalibrator::computeMedian(const std::vector<std::vector<double>>& samples, size_t idx) const {
    std::vector<double> v; for(const auto& s:samples) v.push_back(s[idx]);
    std::sort(v.begin(),v.end());
    return v[v.size()/2];
}

std::pair<double,double> MCMCCalibrator::computeCredibleInterval(const std::vector<std::vector<double>>& samples, size_t idx, double level) const {
    std::vector<double> v; for(const auto& s:samples) v.push_back(s[idx]);
    std::sort(v.begin(),v.end());
    double alpha = 1.0 - level;
    size_t l = static_cast<size_t>(alpha/2.0*v.size());
    size_t u = static_cast<size_t>((1.0-alpha/2.0)*v.size());
    return {v[l],v[u]};
}
