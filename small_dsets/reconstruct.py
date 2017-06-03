#!/usr/bin/env python3

import pandas
import argparse
#from sklearn.preprocessing import LabelEncoder,StandardScaler
import shelve
import tensorflow as tf

def reconstruct(input_file,shelve_file):
    dataset=pandas.read_csv(input_file)
    metadata=shelve.open(shelve_file)
    for column in dataset.columns:
        if column not in metadata["columns"]:
            dataset=dataset.drop(column,axis=1)
    dataset=reconstruct_scale(dataset,metadata["scaler"])
    dataset=reconstruct_labels(dataset,metadata["labels"])
    metadata.close()
    print(dataset[:5])

def reconstruct_scale(dataset,scaler):
    cols=dataset.columns
    return pandas.DataFrame(scaler.inverse_transform(dataset),columns=cols)

def reconstruct_labels(dataset,labels):
    for i in range(len(labels)):
        encoder=labels[i]
        if encoder != None:
            dataset[dataset.columns[i]]=labels[i].inverse_transform(dataset[dataset.columns[i]].apply(int))
    return dataset

def load_model(session,model_file):
    
    

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-i",help="input csv file path.")
    parser.add_argument("-s",help="metadata file.")
    args=parser.parse_args()
    reconstruct(args.i,args.s)

main()
