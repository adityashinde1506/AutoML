# AutoML
## A python based auto experimentation framework for testing your machine learning models

Example script file:

```python
#!/usr/bin/env python3

import sys
sys.path.append(<path_to_AutoML>) # not necesssary if you put AutoML in the python path

import os
import logging
import pandas
from mllib.dataset import *
from mllib.experiment import *

# imports for different models.
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor,BernoulliRBM
from sklearn.metrics import mean_absolute_error


logging.basicConfig(level=logging.DEBUG,filename="griffin.txt")

DATASET_DIR="/home/adityas/Kaggle/datasets/" 

ATTRS=["attr1", ...
        "attr2"]

REMOVE=["attr_remove",...
        "attrn_remove"]

files= list(files) # dataset files list with complete path.

def custom_function(frame):
    return pandas.DataFrame() # Should always return a dataframe object.

MetaExperiment([Experiment(model=<scikit-learn model 1>,
                                datasource=Datasource(files=files,delimiter=" ",
                                    headers=ATTRS,
                                    remove_cols=REMOVE,
                                    rolls=[(<column_to_roll>,-4)],
                                    target_col=<target_atribute>),
                                metric=mean_absolute_error,trials=10,
                                name=<name_of_experiment1>),
Experiment(
                                model=<scikit-learn model 2>,
                                datasource=Datasource(files=files,delimiter=" ",
                                    headers=ATTRS,
                                    remove_cols=REMOVE,
                                    transforms=[custom_function], # custom transform to apply on dataframe.
                                    rolls=[(<column_to_roll>,-96)],
                                    target_col=<target_attribute>),
                                metric=mean_absolute_error,trials=10,
                                name=<name_of_experiment_2>)
])
```