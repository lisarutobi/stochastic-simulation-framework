#pragma once

#include <string>
#include <map>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace stochastic {

/**
 * @brief Classe de base pour les paramètres des processus stochastiques
 * 
 * Design Pattern: Cette classe utilise un map<string, double> flexible
 * plutôt que des attributs fixes, permettant d'ajouter facilement de nouveaux processus
 */
class ProcessParameters {
protected:
    std::map<std::string, double> params_;  // Stockage clé-valeur des paramètres
    
public:
    ProcessParameters() = default;
    virtual ~ProcessParameters() = default;
    
    // Setters/Getters génériques
    void setParameter(const std::string& name, double value) {
        params_[name] = value;
    }
    
    double getParameter(const std::string& name) const {
        auto it = params_.find(name);
        if (it == params_.end()) {
            throw std::runtime_error("Parameter '" + name + "' not found");
        }
        return it->second;
    }
    
    bool hasParameter(const std::string& name) const {
        return params_.find(name) != params_.end();
    }
    
    // Validation abstraite (à implémenter par chaque processus)
    virtual void validate() const = 0;
    
    // Affichage
    virtual std::string toString() const {
        std::string result = "Parameters:\n";
        for (const auto& [key, value] : params_) {
            result += "  " + key + " = " + std::to_string(value) + "\n";
        }
        return result;
    }
    
    // Accès à tous les paramètres
    const std::map<std::string, double>& getAllParameters() const {
        return params_;
    }
};

// paramètres spécifiques pour différents processus stochastiques
/**
 * @brief Paramètres pour Geometric Brownian Motion (GBM)
 * SDE: dS_t = μ*S_t*dt + σ*S_t*dW_t
 */
class GBMParameters : public ProcessParameters {
public:
    GBMParameters(double mu = 0.05, double sigma = 0.2) {
        params_["mu"] = mu;        // Drift (taux de rendement espéré)
        params_["sigma"] = sigma;  // Volatilité
    }
    
    void validate() const override {
        if (params_.at("sigma") <= 0) {
            throw std::invalid_argument("Sigma must be positive");
        }
        // Mu peut être négatif (décroissance)
    }
    
    // Accesseurs spécifiques (plus pratiques que getParameter)
    double getMu() const { return params_.at("mu"); }
    double getSigma() const { return params_.at("sigma"); }
    
    void setMu(double mu) { params_["mu"] = mu; }
    void setSigma(double sigma) { 
        if (sigma <= 0) throw std::invalid_argument("Sigma must be positive");
        params_["sigma"] = sigma; 
    }
};

/**
 * @brief Paramètres pour Ornstein-Uhlenbeck (OU)
 * SDE: dX_t = θ(μ - X_t)*dt + σ*dW_t
 * Processus mean-reverting : revient vers μ avec vitesse θ
 */
class OUParameters : public ProcessParameters {
public:
    OUParameters(double theta = 1.0, double mu = 0.0, double sigma = 0.1) {
        params_["theta"] = theta;  // Vitesse de mean-reversion
        params_["mu"] = mu;        // Niveau moyen (long-term mean)
        params_["sigma"] = sigma;  // Volatilité
    }
    
    void validate() const override {
        if (params_.at("theta") <= 0) {
            throw std::invalid_argument("Theta (mean-reversion speed) must be positive");
        }
        if (params_.at("sigma") <= 0) {
            throw std::invalid_argument("Sigma must be positive");
        }
        // Mu peut être négatif
    }
    
    double getTheta() const { return params_.at("theta"); }
    double getMu() const { return params_.at("mu"); }
    double getSigma() const { return params_.at("sigma"); }
    
    // Half-life de mean-reversion, i.e temps pour parcourir 50% de la distance vers μ
    double getHalfLife() const { 
        return std::log(2.0) / params_.at("theta"); 
    }
};

/**
 * @brief Paramètres pour le modèle de Heston (volatilité stochastique)
 * SDE: 
 *   dS_t = μ*S_t*dt + √v_t*S_t*dW_t^S
 *   dv_t = κ(θ - v_t)*dt + ξ*√v_t*dW_t^v
 * Corrélation entre W^S et W^v = ρ
 */
class HestonParameters : public ProcessParameters {
public:
    HestonParameters(double mu = 0.05, double kappa = 2.0, double theta = 0.04,
                     double xi = 0.3, double rho = -0.7, double v0 = 0.04) {
        params_["mu"] = mu;        // Drift du prix
        params_["kappa"] = kappa;  // Mean-reversion de la variance
        params_["theta"] = theta;  // Variance long-terme
        params_["xi"] = xi;        // Volatilité de la volatilité (vol of vol)
        params_["rho"] = rho;      // Corrélation prix-volatilité
        params_["v0"] = v0;        // Variance initiale
    }
    
    void validate() const override {
        if (params_.at("kappa") <= 0) {
            throw std::invalid_argument("Kappa must be positive");
        }
        if (params_.at("theta") <= 0) {
            throw std::invalid_argument("Theta (long-term variance) must be positive");
        }
        if (params_.at("xi") <= 0) {
            throw std::invalid_argument("Xi (vol of vol) must be positive");
        }
        if (std::abs(params_.at("rho")) > 1.0) {
            throw std::invalid_argument("Rho (correlation) must be in [-1, 1]");
        }
        if (params_.at("v0") <= 0) {
            throw std::invalid_argument("Initial variance v0 must be positive");
        }
        
        // Condition de Feller garantit que la variance reste positive
        // Condition 2*kappa*theta > xi^2
        double kappa = params_.at("kappa");
        double theta = params_.at("theta");
        double xi = params_.at("xi");
        if (2 * kappa * theta <= xi * xi) {
            throw std::invalid_argument(
                "Condition de Feller non respectée, 2*kappa*theta > xi^2 required for positive variance"
            );
        }
    }
    
    double getMu() const { return params_.at("mu"); }
    double getKappa() const { return params_.at("kappa"); }
    double getTheta() const { return params_.at("theta"); }
    double getXi() const { return params_.at("xi"); }
    double getRho() const { return params_.at("rho"); }
    double getV0() const { return params_.at("v0"); }
};

/**
 * @brief Paramètres pour Cox-Ingersoll-Ross (CIR)
 * SDE: dr_t = κ(θ - r_t)*dt + σ*√r_t*dW_t
 * Utilisé pour modéliser les taux d'intérêt
 */
class CIRParameters : public ProcessParameters {
public:
    CIRParameters(double kappa = 0.5, double theta = 0.03, double sigma = 0.1) {
        params_["kappa"] = kappa;  // Mean-reversion speed
        params_["theta"] = theta;  // Long-term mean
        params_["sigma"] = sigma;  // Volatilité
    }
    
    void validate() const override {
        if (params_.at("kappa") <= 0) {
            throw std::invalid_argument("Kappa must be positive");
        }
        if (params_.at("theta") <= 0) {
            throw std::invalid_argument("Theta must be positive");
        }
        if (params_.at("sigma") <= 0) {
            throw std::invalid_argument("Sigma must be positive");
        }
        
        // Condition de Feller pour CIR (similaire à Heston)
        double kappa = params_.at("kappa");
        double theta = params_.at("theta");
        double sigma = params_.at("sigma");
        if (2 * kappa * theta <= sigma * sigma) {
            throw std::invalid_argument(
                "Feller condition violated: 2*kappa*theta > sigma^2 required"
            );
        }
    }
    
    double getKappa() const { return params_.at("kappa"); }
    double getTheta() const { return params_.at("theta"); }
    double getSigma() const { return params_.at("sigma"); }
};

/**
 * @brief Paramètres pour Jump-Diffusion (Merton)
 * SDE: dS_t = μ*S_t*dt + σ*S_t*dW_t + S_t*dJ_t
 * où J_t est un processus de Poisson composé
 */
class LevyJumpParameters : public ProcessParameters {
public:
    LevyJumpParameters(double mu = 0.05, double sigma = 0.2, 
                      double lambda = 0.1, double jumpMean = -0.05, 
                      double jumpStd = 0.1) {
        params_["mu"] = mu;              // Drift
        params_["sigma"] = sigma;        // Volatilité diffusion
        params_["lambda"] = lambda;      // Intensité des sauts (nombre/an)
        params_["jumpMean"] = jumpMean;  // Taille moyenne des sauts
        params_["jumpStd"] = jumpStd;    // Écart-type des sauts
    }
    
    void validate() const override {
        if (params_.at("sigma") <= 0) {
            throw std::invalid_argument("Sigma must be positive");
        }
        if (params_.at("lambda") < 0) {
            throw std::invalid_argument("Lambda (jump intensity) must be non-negative");
        }
        if (params_.at("jumpStd") <= 0) {
            throw std::invalid_argument("Jump std must be positive");
        }
    }
    
    double getMu() const { return params_.at("mu"); }
    double getSigma() const { return params_.at("sigma"); }
    double getLambda() const { return params_.at("lambda"); }
    double getJumpMean() const { return params_.at("jumpMean"); }
    double getJumpStd() const { return params_.at("jumpStd"); }
};

/**
 * @brief Paramètres pour Fractional Brownian Motion (fBM)
 * Processus non-markovien avec mémoire longue
 */
class FBMParameters : public ProcessParameters {
public:
    FBMParameters(double hurst = 0.7, double sigma = 0.1) {
        params_["hurst"] = hurst;  // Paramètre de Hurst ∈ (0,1)
        params_["sigma"] = sigma;  // Volatilité
    }
    
    void validate() const override {
        double H = params_.at("hurst");
        if (H <= 0 || H >= 1.0) {
            throw std::invalid_argument("Hurst parameter must be in (0, 1)");
        }
        if (params_.at("sigma") <= 0) {
            throw std::invalid_argument("Sigma must be positive");
        }
    }
    
    double getHurst() const { return params_.at("hurst"); }
    double getSigma() const { return params_.at("sigma"); }
    
    // Propriétés du processus
    bool isAntiPersistent() const { return params_.at("hurst") < 0.5; }
    bool isPersistent() const { return params_.at("hurst") > 0.5; }
    bool isBrownian() const { return std::abs(params_.at("hurst") - 0.5) < 1e-6; }
};

} // namespace stochastic