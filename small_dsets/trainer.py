#!/usr/bin/env python3

import argparse
import tensorflow as tf
import pandas
import numpy
import importlib
from lib import helpers
from lib import losses
from lib.dataops import DSetHandler

def load_dataset(input_csv):
    dset=DSetHandler()
    X,target=dset.get_training_set(input_csv)
    return X,target

def train(generator,epochs,model,save_dir):
    loss=losses.Loss()
    estopper=helpers.EarlyStopper()
    trn_progress=helpers.TrProgress()
    predictions=model.compute_predictions(model.x)
    cost=loss.mse_loss(predictions,model.y_,batch_size)
    optimizer=tf.train.AdagradOptimizer(1.0).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        epoch=0
        while epoch <= epochs:
            x_train,y_train=next(generator)
            out,_cost=sess.run([optimizer,cost],feed_dict={model.x:x_train,model.y_:y_train})
            if trn_progress.check_progress(_cost,epoch):
                break
            epoch+=1
        saver.save(sess,save_dir)

def main():
    global batch_size
    parser=argparse.ArgumentParser()
    parser.add_argument("-i",help="input csv path.")
    parser.add_argument("-e",help="number of epochs.")
    parser.add_argument("-b",help="batch size.")
    parser.add_argument("-m",help="model file.")
    parser.add_argument("-s",help="path to save trained model.")
    args=parser.parse_args()
    batch_size=int(args.b)
    X,y=load_dataset(args.i)
    global model
    model=importlib.import_module(args.m)
    generator=helpers.batch_generator(X,y,batch_size)
    train(generator,int(args.e),model,args.s)
    

main()
