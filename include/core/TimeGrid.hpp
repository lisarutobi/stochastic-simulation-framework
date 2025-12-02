#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>

namespace stochastic {

/**
 * @brief Gestion du maillage temporel pour la discrétisation des SDE
 * 
 * Cette classe gère la grille de temps pour la simulation :
 * - t0, t1, t2, ..., tN
 * - Avec des pas de temps réguliers ou irréguliers
 * 
 * Exemple : Pour simuler 1 an avec 252 pas (jours de trading) :
 *   TimeGrid grid(0.0, 1.0, 252);
 */
class TimeGrid {
private:
    double t0_;          // Temps initial
    double T_;           // Temps final
    size_t nSteps_;      // Nombre de pas de temps
    double dt_;          // Pas de temps constant
    std::vector<double> times_;  // Grille complète [t0, t1, ..., tN]
    
public:
    // ========================================================================
    // CONSTRUCTEURS
    // ========================================================================
    
    /**
     * @brief Constructeur avec pas de temps régulier
     * @param t0 Temps initial (généralement 0.0)
     * @param T Temps final (en années, par exemple 1.0 = 1 an)
     * @param nSteps Nombre de pas de temps
     * 
     * Exemple : TimeGrid(0.0, 1.0, 252) → dt = 1/252 ≈ 0.004 (1 jour de trading)
     */
    TimeGrid(double t0, double T, size_t nSteps) 
        : t0_(t0), T_(T), nSteps_(nSteps) {
        
        if (T <= t0) {
            throw std::invalid_argument("Final time T must be greater than initial time t0");
        }
        if (nSteps == 0) {
            throw std::invalid_argument("Number of steps must be positive");
        }
        
        dt_ = (T - t0) / nSteps;
        
        // Construire la grille
        times_.reserve(nSteps_ + 1);
        for (size_t i = 0; i <= nSteps_; ++i) {
            times_.push_back(t0_ + i * dt_);
        }
    }
    
    /**
     * @brief Constructeur avec grille personnalisée
     * @param times Vecteur de temps personnalisé (doit être croissant)
     */
    explicit TimeGrid(const std::vector<double>& times) : times_(times) {
        if (times.empty()) {
            throw std::invalid_argument("Time grid cannot be empty");
        }
        
        // Vérifier que la grille est strictement croissante
        for (size_t i = 1; i < times.size(); ++i) {
            if (times[i] <= times[i-1]) {
                throw std::invalid_argument("Time grid must be strictly increasing");
            }
        }
        
        t0_ = times.front();
        T_ = times.back();
        nSteps_ = times.size() - 1;
        
        // Pour une grille irrégulière, dt_ = pas moyen
        dt_ = (T_ - t0_) / nSteps_;
    }
    
    // ========================================================================
    // ACCESSEURS
    // ========================================================================
    
    double getInitialTime() const { return t0_; }
    double getFinalTime() const { return T_; }
    double getTimeStep() const { return dt_; }
    size_t getNumSteps() const { return nSteps_; }
    size_t size() const { return times_.size(); }
    
    /**
     * @brief Accès au temps à l'index i
     * @param i Index dans la grille (0 <= i <= nSteps)
     */
    double operator[](size_t i) const {
        if (i >= times_.size()) {
            throw std::out_of_range("Time index out of range");
        }
        return times_[i];
    }
    
    /**
     * @brief Obtenir l'incrément de temps entre i et i+1
     */
    double getIncrement(size_t i) const {
        if (i >= nSteps_) {
            throw std::out_of_range("Increment index out of range");
        }
        return times_[i+1] - times_[i];
    }
    
    /**
     * @brief Accès à la grille complète
     */
    const std::vector<double>& getTimes() const { return times_; }
    
    // ========================================================================
    // MÉTHODES UTILES
    // ========================================================================
    
    /**
     * @brief Vérifier si la grille est régulière (pas constant)
     */
    bool isRegular() const {
        if (nSteps_ <= 1) return true;
        
        double firstDt = times_[1] - times_[0];
        constexpr double tolerance = 1e-10;
        
        for (size_t i = 1; i < nSteps_; ++i) {
            double currentDt = times_[i+1] - times_[i];
            if (std::abs(currentDt - firstDt) > tolerance) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * @brief Trouver l'index correspondant au temps le plus proche de t
     */
    size_t findNearestIndex(double t) const {
        if (t <= t0_) return 0;
        if (t >= T_) return nSteps_;
        
        // Recherche dichotomique
        size_t left = 0;
        size_t right = times_.size() - 1;
        
        while (right - left > 1) {
            size_t mid = (left + right) / 2;
            if (times_[mid] < t) {
                left = mid;
            } else {
                right = mid;
            }
        }
        
        // Retourner l'index le plus proche
        if (std::abs(times_[left] - t) < std::abs(times_[right] - t)) {
            return left;
        }
        return right;
    }
    
    /**
     * @brief Affichage de la grille (pour debug)
     */
    std::string toString() const {
        std::string result = "TimeGrid:\n";
        result += "  t0 = " + std::to_string(t0_) + "\n";
        result += "  T = " + std::to_string(T_) + "\n";
        result += "  nSteps = " + std::to_string(nSteps_) + "\n";
        result += "  dt = " + std::to_string(dt_) + "\n";
        result += "  Regular: " + std::string(isRegular() ? "Yes" : "No") + "\n";
        return result;
    }
    
    // ========================================================================
    // FACTORY METHODS (méthodes statiques utiles)
    // ========================================================================
    
    /**
     * @brief Créer une grille de trading (252 jours/an)
     * @param nYears Nombre d'années
     */
    static TimeGrid tradingDaysGrid(double nYears) {
        size_t nSteps = static_cast<size_t>(252 * nYears);
        return TimeGrid(0.0, nYears, nSteps);
    }
    
    /**
     * @brief Créer une grille calendaire (365 jours/an)
     */
    static TimeGrid calendarDaysGrid(double nYears) {
        size_t nSteps = static_cast<size_t>(365 * nYears);
        return TimeGrid(0.0, nYears, nSteps);
    }
    
    /**
     * @brief Créer une grille mensuelle
     */
    static TimeGrid monthlyGrid(double nYears) {
        size_t nSteps = static_cast<size_t>(12 * nYears);
        return TimeGrid(0.0, nYears, nSteps);
    }
};

} 