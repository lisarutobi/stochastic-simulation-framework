#ifndef PROCESSUS_STOCHASTIQUE_H
#define PROCESSUS_STOCHASTIQUE_H

#include <vector>
#include <random>

/**
 * Classe abstraite de base pour tous les processus stochastiques
 */
class ProcessusStochastique {
protected:
    double currentState;      // État courant du processus
    double dt;                // Pas de temps
    std::mt19937 rng;         // Générateur de nombres aléatoires
    std::normal_distribution<double> normalDist;
    
public:
    ProcessusStochastique(double initialState, double timeStep, unsigned int seed = 42);
    virtual ~ProcessusStochastique() = default;
    
    // Méthodes virtuelles pure (abstract)
    virtual double nextStep() = 0;
    virtual std::vector<double> simulatePath(int nSteps) = 0;
    virtual std::string getName() const = 0;
    
    // Méthodes communes
    void reset(double initialState);
    double getCurrentState() const;
    double getTimeStep() const;
    
protected:
    double generateNormal(double mean = 0.0, double stddev = 1.0);
};

#endif // PROCESSUS_STOCHASTIQUE_H