#!/usr/bin/env python3

import argparse
import pandas
from reconstruct import Reconstructor
from dataops import DSetHandler

parser=argparse.ArgumentParser()
parser.add_argument("-t",help="path to test file.")
parser.add_argument("-r",help="path to results file.")
parser.add_argument("-m",help="path metadata file.")
args=parser.parse_args()

rec=Reconstructor()
dset=DSetHandler()

ids,scaler,labels=dset.read_meta_file(args.m)
results=pandas.read_csv(args.r)["results"]
test=pandas.read_csv(args.t)
test=test.drop(test.columns[0],axis=1)
test["results"]=results
test=rec.reconstruct_scale(test,scaler)
test=rec.reconstruct_labels(test,labels)
test["results"]=test["results"].apply(int)
print(test[:20])
