# Simulation Stochastique

Projet C++ pour simuler des processus stochastiques et les calibrer sur données de marché.

## Build

```bash
brew install cmake curl nlohmann-json eigen
mkdir build && cd build
cmake .. && make
```

## Modèles

- GBM (Geometric Brownian Motion)
- Heston (volatilité stochastique)
- Ornstein-Uhlenbeck
- CIR (taux d'intérêt)
- Merton (jump-diffusion)
- fBM (mouvement brownien fractionnaire)

## Usage

```cpp
auto gbm = std::make_shared<GeometricBrownianMotion>(100.0, 0.05, 0.2);

MonteCarloEngine::SimulationConfig cfg;
cfg.nPaths = 10000;
cfg.nSteps = 252;
cfg.T = 1.0;

MonteCarloEngine engine(gbm, cfg);
engine.simulate();
std::cout << engine.getStatistics().terminalMean << std::endl;
```

## Calibration

Télécharge les données Yahoo Finance et calibre via MCMC :

```cpp
auto data = YahooFinanceAPI::downloadHistoricalData("AAPL", "2020-01-01", "2024-01-01", "1d");

MCMCCalibrator calibrator(process, tsData, config);
auto result = calibrator.calibrateFromMarketData();
```

## Tests

```bash
./tests/testProcesses
./tests/testMCMCCalibrator
```
