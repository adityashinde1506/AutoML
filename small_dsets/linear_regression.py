#!/usr/bin/env python3

import argparse
import tensorflow as tf
import pandas
import numpy

x=tf.placeholder(tf.float64,[None,290])
y_=tf.placeholder(tf.float64)
batch_size=1000

def load_dataset(input_csv):
    data=pandas.read_csv(input_csv)
    target=data[data.columns[-1]].as_matrix().astype(numpy.float64)
    X=data.drop(data.columns[-1],axis=1).as_matrix().astype(numpy.float64)
    return X,target

def batch_generator(X,y,size=1000):
    i=0
    while True:
        if i > X.shape[0]:
            i=0
        batch_x=X[i:i+size]
        batch_y=y[i:i+size]
        if batch_x.shape[0]==size:
            yield batch_x,batch_y
        else:
            difference=size-batch_x.shape[0]
            batch_x=numpy.vstack((batch_x,X[:difference]))
            batch_y=numpy.hstack((batch_y,y[:difference]))
            yield batch_x,batch_y
        i+=size

def compute_predictions(x):
    W=tf.Variable(tf.random_normal([290,1],mean=0.0,stddev=1.0,dtype=tf.float64))
    b=tf.Variable(tf.random_normal([1],dtype=tf.float64))
    return tf.add(tf.matmul(x,W),b)

def compute_loss(predictions,samples):
    return tf.reduce_sum(tf.pow(predictions-y_,2))/(2.0*samples)

def train(generator,epochs):
    predictions=compute_predictions(x)
    cost=compute_loss(predictions,batch_size)
    optimizer=tf.train.AdagradOptimizer(0.1).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch=0
        loss=0
        while epoch <= epochs:
            x_train,y_train=next(generator)
            out,_cost=sess.run([optimizer,cost],feed_dict={x:x_train,y_:y_train})
            loss+=_cost
            if epoch%100 == 0 and epoch!=0: 
                print("Epoch: %d Loss: %s "%(epoch,str(loss/100.0)))
                loss=0
            epoch+=1
        print()

def main():
    global batch_size
    parser=argparse.ArgumentParser()
    parser.add_argument("-i",help="input csv path.")
    parser.add_argument("-e",help="number of epochs.")
    parser.add_argument("-b",help="batch size.")
    args=parser.parse_args()
    batch_size=int(args.b)
    X,y=load_dataset(args.i)
    generator=batch_generator(X,y,batch_size)
    train(generator,int(args.e))
    

main()
