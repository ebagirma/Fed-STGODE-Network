# STGODE

This is an implementation of [Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting](https://arxiv.org/abs/2106.12931)

install the nightly version of flower by:

```
pip3 install -U flwr-nightly
pip3 install -U flwr-nightly[simulation]
```

version of flower should be **1.4.0**

* use all this code inside this folder
* make sure to activate your environment
* use fed_stgode.py to run the code, change every hyperparameter there and do not touch other files
* use this same file to change the dataset, i am currently using pems04

fed_stgode.py will dump history of training in pickle format by name "flower_history.pkl"

play with this pickle file to see all the result for each metrics for each client, it has more results than the data its trained on

to plot the data of this file use plots_flwr_history.ipynb

* this code spawns different processes in parallel because of framework, ray
* use this code to kill all the processes

  * ```
     kill $(ps aux | grep python | grep -v grep | awk '{print $2}') 
    ```


## Requirements

* python 3.7
* torch 1.7.0+cu101
* torchdiffeq 0.2.2
* fastdtw 0.3.4

## Dataset

The datasets used in our paper are collected by the Caltrans Performance Measurement System(PeMS). Please refer to [STSGCN (AAAI2020)](https://github.com/Davidham3/STSGCN) for the download url.

## Reference

Please cite our paper if you use the model in your own work:

```
@inproceedings{fang2021spatial,
  title={Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting},
  author={Fang, Zheng and Long, Qingqing and Song, Guojie and Xie, Kunqing},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={364--373},
  year={2021}
}
```
