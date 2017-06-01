#!/usr/bin/env python3

import pandas
import argparse
from sklearn.preprocessing import LabelEncoder,StandardScaler
import shelve


def get_column_dtypes(dataframe):
    return list(map(lambda x:dataframe[x].dtype,dataframe.columns))

def convert_to_numeric(dframe):
    dtypes=get_column_dtypes(dframe)
    label_transform=list()
    for i in range(len(dtypes)):
        if dtypes[i]=="object":
            ltransform=LabelEncoder()
            dframe[dframe.columns[i]]=ltransform.fit_transform(dframe[dframe.columns[i]])
            label_transform.append(ltransform)
        else:
            label_transform.append(None)
    return dframe,label_transform

def rescale(dframe):
    scaler=StandardScaler()
    dframe=pandas.DataFrame(scaler.fit_transform(dframe),columns=dframe.columns)
    return dframe,scaler

def write_data(dframe,labels,scaler,output):
    dframe.to_csv(output+"/data.csv")
    metadata=shelve.open(output+"/metadata")
    metadata["columns"]=dframe.columns
    metadata["labels"]=labels
    metadata["scaler"]=scaler
    metadata.close()

def preprocess_data(input_path,output_path):
    dframe=read_data(input_path).fillna(0)
    ids=dframe["id"]
    dframe=dframe.drop("id",axis=1).drop("timestamp",axis=1)
    dframe,labels=convert_to_numeric(dframe)
    dframe,scaler=rescale(dframe)
    write_data(dframe,labels,scaler,output_path)

def read_data(input_path):
    return pandas.read_csv(input_path)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-i",help="input csv file path.")
    parser.add_argument("-o",help="path to place preprocessed csv.")
    args=parser.parse_args()
    preprocess_data(args.i,args.o)

main()
