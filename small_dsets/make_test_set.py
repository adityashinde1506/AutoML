#!/usr/bin/env python3

import argparse
from lib.dataops import DSetHandler

def run():
    dset=DSetHandler()
    dataframe=dset.import_from_csv(args.i)
    dataframe=dataframe.drop(dataframe.columns[0],axis=1)
    dataframe["dummy"]=0
    dataframe=dset.apply_meta(dataframe,args.m)
    dataframe=dataframe.drop("dummy",axis=1)
    dataframe.to_csv(args.o+"test.csv")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-i",help="input csv path.")
    parser.add_argument("-m",help="path to metadata file.")
    parser.add_argument("-o",help="path to store output.")
    global args
    args=parser.parse_args()
    run()

main()
