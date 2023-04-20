# Source Code (src)

This directory contains the source code for the federated learning models and the Flower framework implementations. The code is organized into the following subdirectories:

## federated_learning

This subdirectory contains the implementation of the federated learning models without the Flower framework:

- `stg_ode.py`: Contains the implementation of the Federated STG ODE model.
- `gru.py`: Contains the implementation of the Federated Gated Recurrent Unit (GRU) model.
- `lstm.py`: Contains the implementation of the Federated Long Short-Term Memory (LSTM) model.

## flower_framework

This subdirectory contains the implementation of the federated learning models using the Flower framework:

- `stg_ode_flower.py`: Contains the implementation of the Federated STG ODE model with Flower.
- `gru_flower.py`: Contains the implementation of the Federated Gated Recurrent Unit (GRU) model with Flower.
- `lstm_flower.py`: Contains the implementation of the Federated Long Short-Term Memory (LSTM) model with Flower.

## utils

This subdirectory contains utility functions and helper code:

- `helpers.py`: Contains various helper functions used throughout the project.
