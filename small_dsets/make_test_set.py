#!/usr/bin/env python3

import argparse
import pandas
from dataops import DSetHandler

def run():
    dset=DSetHandler()
    dataframe=dset.import_from_csv(args.i)
    dataframe["results"]=0
    dataframe=dataframe.drop(dataframe.columns[0],axis=1)
    dataframe=dset.apply_meta(dataframe,args.m)
    dataframe=dataframe.drop("results",axis=1)
    dataframe.to_csv(args.o+"test.csv")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-i",help="input path to testing csv.")
    parser.add_argument("-m",help="path to metadata file.")
    parser.add_argument("-o",help="output path.")
    global args
    args=parser.parse_args()
    run()

main()
