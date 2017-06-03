#!/usr/bin/env python3

import tensorflow as tf
import pandas
import argparse

def load_model(session,model_file):
    _file=model_file
    _dir="/".join(model_file.split("/")[:-1])+"/"
    saver=tf.train.import_meta_graph(_file)
    saver.restore(session,tf.train.latest_checkpoint(_dir))

def get_data(_file,test=True):
    dataset=pandas.read_csv(_file)
    orignal_data=dataset
    if test:
        targets=dataset[dataset.columns[-1]]
        run_data=dataset.drop(dataset.columns[-1],axis=1)
    else:
        targets=None
        run_data=orignal_data
    return orignal_data,run_data.as_matrix(),targets.as_matrix()

def run():
    ref_data,run_data,labels=get_data(args.i)
    with tf.Session() as sess:
        load_model(sess,args.m)
        sess.run(tf.global_variables_initializer())
        graph=tf.get_default_graph()
        i=graph.get_tensor_by_name("input:0")
        o=graph.get_tensor_by_name("labels:0")
        predictions=graph.get_tensor_by_name("prediction:0")
        result=sess.run(predictions,{i:run_data})
        print(result)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-m",help="model file.")
    parser.add_argument("-i",help="input data file.")
    global args
    args=parser.parse_args()
    run()
    

main()
