#!/usr/bin/env python3

import tensorflow as tf
import argparse
from lib.dataops import DSetHandler
import pandas

def load_model(session,model_file):
    _file=model_file
    _dir="/".join(model_file.split("/")[:-1])+"/"
    saver=tf.train.import_meta_graph(_file)
    saver.restore(session,tf.train.latest_checkpoint(_dir))

def run():
    datahandler=DSetHandler()
    run_data=datahandler.get_testing_set(args.i)
    with tf.Session() as sess:
        load_model(sess,args.m)
        #sess.run(tf.global_variables_initializer())
        graph=tf.get_default_graph()
        i=graph.get_tensor_by_name("input:0")
        predictions=graph.get_tensor_by_name("prediction:0")
        result=sess.run(predictions,{i:run_data})
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-m",help="model file.")
    parser.add_argument("-i",help="input data file.")
    parser.add_argument("-o",help="path to store results.")
    global args
    args=parser.parse_args()
    result=run()
    pandas.DataFrame(result,columns=["results"]).to_csv(args.o+"results.csv")

main()
