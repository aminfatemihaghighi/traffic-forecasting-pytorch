# Spatial-Temporal Traffic Demand Forecasting 🚦📈

## Overview
This repository contains a production-ready **Deep Learning pipeline** built with **PyTorch** to forecast multivariate spatial-temporal traffic volumes across urban sensor networks. 

By predicting future traffic congestion based on historical sequence data, this model serves as the core predictive engine for smart city infrastructure, intelligent traffic light control, and dynamic route planning algorithms.

## Neural Network Architecture & Pipeline
The project moves beyond simple machine learning scripts by implementing a robust, object-oriented PyTorch architecture:

1. **Custom Dataloaders (`torch.utils.data.Dataset`):** * Engineered a custom dataset class to ingest multivariate time-series data from 50+ simulated urban sensors.
   * Implements a sliding-window technique to separate historical sequence lengths (e.g., past 60 minutes) from prediction horizons (e.g., next 15 minutes).

2. **Sequence-to-Sequence LSTM (`nn.Module`):**
   * Utilizes Long Short-Term Memory (LSTM) layers to capture complex temporal dependencies and prevent the vanishing gradient problem inherent in long traffic sequences.
   * Includes Dropout layers to ensure the model generalizes well to unseen data and resists overfitting.

3. **Professional Training Loop:**
   * Features a decoupled, object-oriented training loop using the Adam optimizer and Mean Squared Error (MSE) loss.
   * Integrates the standard Python `logging` library instead of basic print statements to track epoch loss, simulating enterprise-level deployment monitoring.

## Key Technical Implementations
* **Multivariate Forecasting:** Unlike univariate models that predict one road's traffic, this architecture predicts the state of the *entire sensor network* simultaneously.
* **Defensive Engineering:** Incorporates type hinting (`typing.Tuple`, `Optional`), error handling (`try/except`), and explicit memory management during tensor processing.
* **Scalability:** The pipeline is designed to easily swap synthetic data with real-world Parquet/CSV datasets (e.g., PeMS traffic data) without rewriting the core architecture.

## Tech Stack
* **Framework:** PyTorch (`torch`, `torch.nn`, `torch.utils.data`)
* **Language:** Python 3.x
* **Data Processing:** NumPy, pandas
* **Concepts:** Deep Learning, LSTMs, Multivariate Time-Series Analysis
