#!/usr/bin/env python3

import pandas
import argparse
from dataops import DSetHandler

def preprocess_data(input_file,output_path):
    dsethandler=DSetHandler()
    dataframe=dsethandler.import_from_csv(input_file).drop("timestamp",axis=1)
    print(dataframe[:5])
    dsethandler.preprocess_with_labels(dataframe,output_path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-i",help="input csv file path.")
    parser.add_argument("-o",help="path to place preprocessed csv.")
    args=parser.parse_args()
    preprocess_data(args.i,args.o)

main()
