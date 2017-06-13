#!/usr/bin/env python3

import sys

sys.path.append("/home/adityas/Projects/ML_scripts")

import mllib
from mllib.pipeline import Pipeline
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("-s",help="specifications file.")
args=parser.parse_args()

pipeline=Pipeline(args.s)
res=pipeline.train_run()
print(res)
#pipeline.debug()
