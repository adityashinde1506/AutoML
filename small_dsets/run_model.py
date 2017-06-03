#!/usr/bin/env python3

import tensorflow as tf
import argparse
from dataops import DSetHandler

def load_model(session,model_file):
    _file=model_file
    _dir="/".join(model_file.split("/")[:-1])+"/"
    saver=tf.train.import_meta_graph(_file)
    saver.restore(session,tf.train.latest_checkpoint(_dir))

def run():
    datahandler=DSetHandler()
    dset=datahandler.import_from_csv(args.i)
    run_data,targets=datahandler.seperate_labels(dset)
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
