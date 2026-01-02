#pragma once

#include "core/StochasticProcess.hpp"
#include "core/TimeGrid.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

namespace stochastic {

class FractionalBrownianMotion : public StochasticProcess {
private:
    double H_;
    double sigma_;

    // Cache pour la décomposition de Cholesky
    mutable std::vector<std::vector<double>> choleskyL_;
    mutable size_t cachedN_ = 0;
    mutable double cachedDt_ = 0.0;

    // Autocovariance des incréments du fBM
    double incrementAutocovariance(int k, double dt) const {
        double twoH = 2.0 * H_;
        double dtPow = std::pow(dt, twoH);

        if (k == 0) {
            return sigma_ * sigma_ * dtPow;
        }

        double absK = std::abs(k);
        return 0.5 * sigma_ * sigma_ * dtPow * (
            std::pow(absK + 1, twoH)
          - 2.0 * std::pow(absK, twoH)
          + std::pow(absK - 1, twoH)
        );
    }

    // Construction de la décomposition de Cholesky
    void buildCholeskyDecomposition(size_t N, double dt) const {
        if (N == cachedN_ && std::abs(dt - cachedDt_) < 1e-12) {
            return;
        }

        cachedN_ = N;
        cachedDt_ = dt;

        // Construction de la matrice de covariance
        std::vector<std::vector<double>> C(N, std::vector<double>(N, 0.0));

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                C[i][j] = incrementAutocovariance(static_cast<int>(i) - static_cast<int>(j), dt);
                C[j][i] = C[i][j];
            }
        }

        // Décomposition de Cholesky : C = L * L^T
        choleskyL_.assign(N, std::vector<double>(N, 0.0));

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = C[i][j];
                for (size_t k = 0; k < j; ++k) {
                    sum -= choleskyL_[i][k] * choleskyL_[j][k];
                }
                if (i == j) {
                    if (sum <= 0) {
                        choleskyL_[i][i] = 1e-10;
                    } else {
                        choleskyL_[i][i] = std::sqrt(sum);
                    }
                } else {
                    choleskyL_[i][j] = sum / choleskyL_[j][j];
                }
            }
        }
    }

    // Densité spectrale des incréments du fBM
    double spectralDensity(double freq, double dt) const {
        if (std::abs(freq) < 1e-10) return 0.0;

        double CH = std::sin(M_PI * H_) * std::tgamma(2.0 * H_ + 1.0) / M_PI;
        double fBmSpectral = CH * sigma_ * sigma_ * std::pow(std::abs(freq), -(2.0 * H_ + 1.0));
        double transferFunction = 4.0 * std::pow(std::sin(M_PI * freq * dt), 2);

        return fBmSpectral * transferFunction;
    }

public:
    FractionalBrownianMotion(
        double init = 0.0,
        double H = 0.5,
        double sigma = 1.0,
        unsigned long seed = 0
    ) : StochasticProcess(init, seed), H_(H), sigma_(sigma) {
        if (H <= 0.0 || H >= 1.0) {
            throw std::invalid_argument("Hurst parameter H must be in (0, 1)");
        }
        if (sigma <= 0.0) {
            throw std::invalid_argument("Sigma must be positive");
        }
    }

    std::string getName() const override {
        return "Fractional Brownian Motion (H=" + std::to_string(H_) + ")";
    }

    double drift(double, double) const override { return 0.0; }
    double diffusion(double, double) const override { return sigma_; }

    void validateParameters() const override {
        if (H_ <= 0.0 || H_ >= 1.0 || sigma_ <= 0.0) {
            throw std::invalid_argument("Invalid fBM parameters");
        }
    }

    bool hasStationaryDistribution() const override { return false; }
    bool canBeNegative() const override { return true; }

    // Le fBM est non-markovien, donc nextStep n'est pas applicable
    double nextStep(double) override {
        throw std::runtime_error("fBM is non-Markovian - use simulatePath() with full grid");
    }

    // Simulation exacte via décomposition de Cholesky
    std::vector<double> simulatePathExact(const TimeGrid& grid) override {
        size_t N = grid.getNumSteps();
        double dt = grid.getFinalTime() / N;

        buildCholeskyDecomposition(N, dt);

        // Génération de N normales i.i.d.
        std::vector<double> Z(N);
        for (size_t i = 0; i < N; ++i) {
            Z[i] = generateGaussian();
        }

        // Calcul des incréments corrélés : X = L * Z
        std::vector<double> increments(N, 0.0);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                increments[i] += choleskyL_[i][j] * Z[j];
            }
        }

        // Construction du chemin par somme cumulative
        std::vector<double> path(N + 1);
        path[0] = initialState_;
        for (size_t i = 1; i <= N; ++i) {
            path[i] = path[i - 1] + increments[i - 1];
        }

        currentState_ = path.back();
        return path;
    }

    // Simulation approximative (plus rapide pour grands N)
    std::vector<double> simulatePathApproximate(const TimeGrid& grid) {
        size_t N = grid.getNumSteps();
        double dt = grid.getFinalTime() / N;
        double scale = sigma_ * std::pow(dt, H_);

        std::vector<double> path(N + 1);
        path[0] = initialState_;

        for (size_t i = 1; i <= N; ++i) {
            path[i] = path[i - 1] + scale * generateGaussian();
        }

        currentState_ = path.back();
        return path;
    }

    std::vector<double> simulatePath(const TimeGrid& grid, bool useExact = true) override {
        if (useExact && grid.getNumSteps() <= 2000) {
            return simulatePathExact(grid);
        } else {
            return simulatePathApproximate(grid);
        }
    }

    std::vector<double> simulatePath(double T, size_t nSteps, bool useExact = true) override {
        TimeGrid g(0.0, T, nSteps);
        return simulatePath(g, useExact);
    }

    std::shared_ptr<StochasticProcess> clone() const override {
        auto cloned = std::make_shared<FractionalBrownianMotion>(initialState_, H_, sigma_);
        cloned->setSeed(std::random_device{}());
        return cloned;
    }

    std::vector<std::string> getParameterNames() const override {
        return {"H", "sigma"};
    }

    std::vector<double> getParametersVector() const override {
        return {H_, sigma_};
    }

    void setParametersVector(const std::vector<double>& p) override {
        if (p.size() != 2) {
            throw std::invalid_argument("fBM requires 2 parameters: H, sigma");
        }
        H_ = p[0];
        sigma_ = p[1];
        validateParameters();
        cachedN_ = 0;
    }

    // Log-vraisemblance via approximation de Whittle
    double logLikelihood(const std::vector<double>& observations, double dt) const override {
        size_t N = observations.size();
        if (N < 10) return 0.0;

        // Calcul des incréments
        std::vector<double> increments(N - 1);
        for (size_t i = 0; i < N - 1; ++i) {
            increments[i] = observations[i + 1] - observations[i];
        }

        size_t n = increments.size();
        double mean = 0.0;
        for (double x : increments) mean += x;
        mean /= n;

        // Centrage des incréments
        for (double& x : increments) x -= mean;

        // Calcul du périodogramme via DFT simplifiée
        double logLik = 0.0;
        size_t numFreq = n / 2;

        for (size_t k = 1; k <= numFreq; ++k) {
            double freq = static_cast<double>(k) / (n * dt);

            double cosSum = 0.0, sinSum = 0.0;
            for (size_t j = 0; j < n; ++j) {
                double angle = 2.0 * M_PI * k * j / n;
                cosSum += increments[j] * std::cos(angle);
                sinSum += increments[j] * std::sin(angle);
            }
            double periodogram = (cosSum * cosSum + sinSum * sinSum) / n;

            double S = spectralDensity(freq, dt);

            if (S > 1e-15) {
                logLik -= 0.5 * (std::log(S) + periodogram / S);
            }
        }

        logLik -= numFreq * 0.5 * std::log(2.0 * M_PI);

        return logLik;
    }

    // Accesseurs spécifiques
    double getHurst() const { return H_; }
    double getSigma() const { return sigma_; }

    bool isPersistent() const { return H_ > 0.5; }
    bool isAntiPersistent() const { return H_ < 0.5; }

    // Variance théorique au temps t
    double theoreticalVariance(double t) const {
        return sigma_ * sigma_ * std::pow(t, 2.0 * H_);
    }
};

}
