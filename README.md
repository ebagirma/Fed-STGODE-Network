[![Python application](https://github.com/ebagirma/Federated-Learning/actions/workflows/python-app.yml/badge.svg)](https://github.com/ebagirma/Fed-STGODE-Network/actions/workflows/python-app.yml)  [![Flower implination of Stg-Ode In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12ID1n0OsS7d05mVOo8a8kuiZNi7cix2X?usp=sharing)

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

- GRU (Gated Recurrent Unit)
- RNN (Recurrent Neural Network)
- LSTM (Long Short-Term Memory)
- STGODE (Spatio-Temporal Graph ODE Networks)

## Results

After training and evaluating the models, the results will be saved in the `results` directory. You can find the prediction accuracy, loss, and training time for each model.

## Contributing

We welcome contributions to improve the project. Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
