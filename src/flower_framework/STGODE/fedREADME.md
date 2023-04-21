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
