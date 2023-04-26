[![Python application](https://github.com/ebagirma/Federated-Learning/actions/workflows/python-app.yml/badge.svg)](https://github.com/ebagirma/Federated-Learning/actions/workflows/python-app.yml)  [![Flower implination of Stg-Ode In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12ID1n0OsS7d05mVOo8a8kuiZNi7cix2X?usp=sharing) [![Jupyter Notebook Demo](https://img.shields.io/badge/Demo-Jupyter%20Notebook-informational)](./notebooks/Ploting_for_models.ipynb)

# Federated Learning Project: Traffic Flow Prediction with PEMS Dataset

This repository contains the code and resources for our federated learning project, which aims to predict traffic flow using the PEMS (PeMS Traffic Monitoring) dataset. We've implemented various deep learning models, including GRU, RNN, LSTM, and STGODE, to provide an accurate and reliable traffic flow prediction system.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Models](#models)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

Traffic congestion is a growing issue in urban areas, resulting in economic loss and environmental problems. Accurate traffic flow predictions can help traffic management authorities implement effective traffic control and provide route suggestions to travelers. In this project, we use federated learning to train deep learning models while preserving data privacy.

## Installation

To set up the project, follow these steps:

<pre>
git clone https://github.com/ebagirma/Federated-Learning.git && cd Federated-Learning && pip install -r requirements.txt
</pre>



## Dataset

We use the PEMS (PeMS Traffic Monitoring) dataset, which contains real-time traffic data collected from loop detectors on California highways. The dataset includes traffic speed, occupancy, and flow measurements.

you can download the dataset: git clone https://github.com/ebagirma/Pems_Dataset.git

## Models

We've implemented the following deep learning models for traffic flow prediction:

- GRU (Gated Recurrent Unit) [![Jupyter Notebook Demo](https://img.shields.io/badge/Demo-Jupyter%20Notebook-informational)](./notebooks/flwr_gru.ipynb)
- RNN (Recurrent Neural Network) (Long Short-Term Memory) [![Jupyter Notebook Demo](https://img.shields.io/badge/Demo-Jupyter%20Notebook-informational)](./notebooks/Ploting_for_models.ipynb)
- LSTM [![Jupyter Notebook Demo](https://img.shields.io/badge/Demo-Jupyter%20Notebook-informational)](./notebooks/flwr_lstm.ipynb)
- STGODE (Spatio-Temporal Graph ODE Networks)  See the colab file for more details [![Flower implination of Stg-Ode In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12ID1n0OsS7d05mVOo8a8kuiZNi7cix2X?usp=sharing)


## Results

We present the validation results of our models on the PEMS04 dataset. The graph below demonstrates the performance of each model in terms of prediction accuracy.

![Validation results on PEMS04 dataset](https://user-images.githubusercontent.com/48454309/234192878-c74ecb99-2ed4-4503-8864-fef103a34ddd.png)

These results showcase the effectiveness of using federated learning with stg ode models in predicting traffic flow patterns with lower loss. We have also implemented without the framework which you can find it here [![Flower implination of Stg-Ode In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1En9YarnWPlZ7iwlPF8RuSFGtQxGv9H8W?usp=sharing)


## Contributing

We welcome contributions to improve the project. Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/ebagirma/Federated-Learning/blob/main/LICENSE.txt) for more information.

## Credits

This project is built upon the works of various researchers and developers. We would like to express our gratitude for the following repositories:

- [STGODE](https://github.com/square-coder/STGODE)
- [Federated-Learning-PyTorch](https://github.com/vaseline555/Federated-Learning-PyTorch)
- [Flower](https://flower.dev/)
- [FedGRU](https://github.com/Practicing-Federated-Learning-for-IoT/FedGRU)
