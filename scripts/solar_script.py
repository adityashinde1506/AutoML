#!/usr/bin/env python3

import sys
import os
import pandas
import numpy
import matplotlib.pyplot as plt

sys.path.append("/home/adityas/Projects/ML_scripts")

import mllib
from mllib.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("-s",help="specifications file.")
args=parser.parse_args()

pipeline=Pipeline(args.s)

model=MLPRegressor(max_iter=10,warm_start=True)

def plot_graph(featurename,feature,solar,pred):
    plt.scatter(feature[:10],solar[:10],c="red")
    plt.scatter(feature[:10],pred[:10],c="blue")
    plt.xlabel(column)
    plt.ylabel("Solar")
    plt.show()

val_data=pipeline.run_data_section().sample(1000)
val_y=val_data["Solar"].as_matrix()
val_X=val_data.drop("Solar",axis=1).as_matrix()

for column in val_data.columns:
    plot_graph(column,val_data[column],val_data["Solar"],val_data["Solar"])

for i in range(1000):
    data=pipeline.run_data_section()
    y=data["Solar"].as_matrix()
    X=data.drop("Solar",axis=1).as_matrix()
    model.fit(X,y)
    out=model.predict(val_X)
    data=numpy.vstack((val_y,out))
    os.system("clear")
    print(mean_squared_error(val_y,out,multioutput="uniform_average"))
    print(pandas.DataFrame(data=data.T,columns=["true","pred"])[:10])
    

for column in val_data.columns:
    plot_graph(column,val_data[column],val_data["Solar"],model.predict(val_X))


#print(res)

#pipeline.debug()
