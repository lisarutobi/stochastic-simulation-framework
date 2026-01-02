# Simulation Stochastique en C++

Ce projet fournit un **framework C++** pour **simuler des processus stochastiques**, effectuer des **simulations Monte Carlo**, et **calibrer des modèles financiers sur des données de marché réelles**.

Le but principal est de permettre :
- L’étude de trajectoires stochastiques (prix, taux, volatilité)  
- La calibration de modèles via des méthodes Bayésiennes (MCMC)  
- L’export et l’analyse statistique de simulations  

---

## Build / Compilation

Le projet utilise **CMake** et dépend de quelques bibliothèques :

- [Eigen](https://eigen.tuxfamily.org/) pour l’algèbre linéaire  
- [nlohmann/json](https://github.com/nlohmann/json) pour JSON  
- curl (optionnel, pour récupérer les données Yahoo Finance)  

```bash
# Installer les dépendances sur macOS
sudo install cmake curl nlohmann-json eigen

# Créer le dossier build
mkdir build && cd build

# Générer les Makefiles et compiler
cmake ..
make
```

## Modèles stochastiques disponibles
Le framework inclut plusieurs processus financiers classiques :
- GBM (Geometric Brownian Motion)	Mouvement brownien géométrique pour modéliser les prix d’actifs.
- Heston	Modèle à volatilité stochastique.
- Ornstein-Uhlenbeck	Processus de retour à la moyenne (mean-reverting).
- CIR	Cox-Ingersoll-Ross, utilisé pour les taux d’intérêt.
- Merton	Modèle jump-diffusion (sauts discrets).
- fBM	Fractional Brownian Motion, pour corrélations temporelles non-Markoviennes.

Chaque processus dérive de la classe StochasticProcess et implémente :
- nextStep(dt) : évolution sur un pas de temps
- drift(t, x) et diffusion(t, x) : équations de SDE
- clone() : copie polymorphique (nécessaire pour Monte Carlo parallèle)
- logLikelihood() : log-vraisemblance pour calibration MCMC

## Utilisation – Simulation Monte Carlo

Voici un exemple simple pour simuler un GBM :
```cpp
#include "processes/GeometricBrownianMotion.hpp"
#include "simulation/MonteCarloEngine.hpp"

// Création du processus
auto gbm = std::make_shared<GeometricBrownianMotion>(100.0, 0.05, 0.2);

// Configuration Monte Carlo
MonteCarloEngine::SimulationConfig cfg;
cfg.nPaths = 10000;          // nombre de trajectoires
cfg.nSteps = 252;            // nombre de pas par trajectoire
cfg.T = 1.0;                 // horizon en années
cfg.useExactScheme = false;  // Euler si pas d’exact

MonteCarloEngine engine(gbm, cfg);
engine.simulate();

// Statistiques finales
std::cout << "Moyenne finale : " << engine.getStatistics().terminalMean << std::endl;
```

Parallélisation automatique
Le moteur choisit séquentiel ou parallèle selon le nombre de threads et le nombre de trajectoires.
Chaque thread utilise une graine différente pour que les simulations soient indépendantes.

## Calibration MCMC
Pour calibrer un processus sur des données historiques :
```cpp
#include "calibration/MCMCCalibrator.hpp"
#include "data/YahooFinanceAPI.hpp"

// Récupération des données Yahoo Finance
auto tsData = YahooFinanceAPI::downloadHistoricalData(
    "AAPL", "2020-01-01", "2024-01-01", "1d"
);

// Création du calibrateur
MCMCCalibrator calibrator(process, tsData, config);

// Lancement de la calibration
auto result = calibrator.calibrateFromMarketData();

// Accès aux résultats
std::cout << "Paramètre moyen : " << result.meanParams["sigma"] << std::endl;
std::cout << "Intervalle crédible 95% : "
          << result.credibleIntervals95["sigma"].first << " - "
          << result.credibleIntervals95["sigma"].second
          << std::endl;
```

Caractéristiques :
- Algorithme Metropolis-Hastings avec covariance adaptative
- Support pour priors uniforme, normal et log-normal
- Compatible avec tout processus dérivant de StochasticProcess

## Tests avec Catch2

```bash
./tests/testProcesses
./tests/testMCMCCalibrator
./tests/testDataPreprocessor
./tests/testMonteCarloEngine
./tests/testProcesss
./tests/testTimeGrid
./tests/testYahooFinance
```
## Structure du projet
stochastic-simulation-framework/
├─ include/
│  ├─ core/           # Classes de base : StochasticProcess.hpp, TimeGrid.hpp, ProcessParameters.hpp
|  ├─ data/           # DataPreprocessor.hpp, MarketDataLoader.hpp, TimeSerisData.hpp, YahooFinanceApi.hpp
│  ├─ processes/      # GBM, Heston, OU, CIR, Levy, fBM.
|  ├─ simulation/     # MonteCarloEngine.hpp
│  └─ calibration/    # MCMCCalibrator.hpp
├─ src/
│  ├─ simulation/     # MonteCarloEngine.cpp
│  ├─ data/           # DataPreprocessor.cpp, MarketDataLoader.cpp, YahooFinanceApi.cpp
│  └─ calibration/    # MCMCCalibrator.cpp
├─ tests/             # Tests unitaires Catch2
├─ CMakeLists.txt
└─ README.md