# Framework de Simulation Stochastique

Framework C++ pour la simulation et la calibration de processus stochastiques en finance quantitative.

## Fonctionnalités

- **Processus stochastiques** : GBM, Heston, Ornstein-Uhlenbeck, CIR, Merton Jump-Diffusion, fBM
- **Simulation Monte Carlo** : Euler-Maruyama, schémas exacts, variables antithétiques
- **Calibration MCMC** : Metropolis-Hastings adaptatif avec différentes lois a priori
- **Données de marché** : Téléchargement Yahoo Finance, prétraitement, calcul des rendements

## Structure du projet

```
├── include/
│   ├── core/           # Classes de base (StochasticProcess, ProcessParameters, TimeGrid)
│   ├── processes/      # Implémentations des processus (GBM, Heston, OU, CIR, Levy, fBM)
│   ├── simulation/     # Moteur Monte Carlo
│   ├── calibration/    # Calibrateur MCMC
│   └── data/           # Chargement et prétraitement des données
├── src/                # Fichiers d'implémentation
├── tests/              # Tests unitaires (Catch2)
└── main.cpp            # Application de démonstration
```

## Compilation

```bash
mkdir build && cd build
cmake ..
make
```

## Dépendances

- C++17
- CMake 3.10+
- CURL (pour Yahoo Finance)
- nlohmann/json
- Eigen3

## Utilisation

```cpp
#include "processes/GeometricBrownianMotion.hpp"
#include "simulation/MonteCarloEngine.hpp"

using namespace stochastic;

// Créer un processus GBM
GeometricBrownianMotion gbm(100.0, 0.05, 0.2);

// Configurer le moteur Monte Carlo
MonteCarloEngine::Config config;
config.numPaths = 10000;
config.numSteps = 252;
config.timeHorizon = 1.0;
config.useAntithetic = true;

MonteCarloEngine engine(config);
auto results = engine.simulate(gbm);

std::cout << "Prix moyen: " << results.statistics.mean << std::endl;
```

## Processus disponibles

| Processus | EDS | Paramètres |
|-----------|-----|------------|
| GBM | dS = μS dt + σS dW | μ, σ |
| Heston | dS = μS dt + √v S dW, dv = κ(θ-v)dt + ξ√v dZ | μ, κ, θ, ξ, ρ |
| OU | dX = θ(μ-X) dt + σ dW | θ, μ, σ |
| CIR | dr = κ(θ-r) dt + σ√r dW | κ, θ, σ |
| Merton | dS/S = (μ-λk) dt + σ dW + dJ | μ, σ, λ, μ_J, σ_J |
| fBM | B_H(t) avec H ∈ (0,1) | H, σ |

## Calibration MCMC

```cpp
#include "calibration/MCMCCalibrator.hpp"

MCMCCalibrator calibrator;
calibrator.setNumIterations(50000);
calibrator.setBurnIn(10000);

// Définir les priors
calibrator.setPrior("mu", MCMCCalibrator::PriorType::NORMAL, 0.0, 0.5);
calibrator.setPrior("sigma", MCMCCalibrator::PriorType::LOGNORMAL, -1.6, 0.5);

auto results = calibrator.calibrate(gbm, marketData, dt);
```

## Licence

Projet académique - Usage libre pour l'éducation et la recherche.
