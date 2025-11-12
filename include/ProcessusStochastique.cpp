#include "ProcessusStochastique.h"
#include <ctime>

ProcessusStochastique::ProcessusStochastique(double initialState, double timeStep, unsigned int seed)
    : currentState(initialState), dt(timeStep), rng(seed), normalDist(0.0, 1.0) {
    if (seed == 0) {
        rng.seed(static_cast<unsigned int>(std::time(nullptr)));
    }
}

void ProcessusStochastique::reset(double initialState) {
    currentState = initialState;
}

double ProcessusStochastique::getCurrentState() const {
    return currentState;
}

double ProcessusStochastique::getTimeStep() const {
    return dt;
}

double ProcessusStochastique::generateNormal(double mean, double stddev) {
    return mean + stddev * normalDist(rng);
}