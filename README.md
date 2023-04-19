[![Python application](https://github.com/ebagirma/Fed-STGODE-Network/actions/workflows/python-app.yml/badge.svg)](https://github.com/ebagirma/Fed-STGODE-Network/actions/workflows/python-app.yml)




## Parameters

Here are the parameters that can be passed to the script:

* `--remote`: whether the code runs on a remote server.
* `--num-gpu`: the number of GPUs to use. Default is 0.
* `--epochs`: the number of training epochs. Default is 10.
* `--batch-size`: the batch size. Default is 16.
* `--batch`: the batch size. Default is 16.
* `--frac`: the fraction of clients (C). Default is 0.01.
* `--num_users`: the number of users. Default is 100.
* `--filename`: the name of the file to load data from. Default is 'pems04'.
* `--train-ratio`: the ratio of training dataset. Default is 0.6.
* `--valid-ratio`: the ratio of validating dataset. Default is 0.2.
* `--his-length`: the length of the history time series of input. Default is 12.
* `--pred-length`: the length of the target time series for prediction. Default is 12.
* `--sigma1`: sigma for the semantic matrix. Default is 0.1.
* `--sigma2`: sigma for the spatial matrix. Default is 10.
* `--thres1`: the threshold for the semantic matrix. Default is 0.6.
* `--thres2`: the threshold for the spatial matrix. Default is 0.5.
* `--lr`: the learning rate. Default is 2e-3.
* `--log`: whether to write log to files.
