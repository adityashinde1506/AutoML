#!/usr/bin/env python3

import time

class EarlyStopper:

    def __init__(self,threshold=0.001):
        self.patience=5
        self.threshold=threshold
        self.window=list()

    def __add_value(self,val):
        if len(self.window)>20:
            self.window.pop()
        self.window.insert(0,val)

    def early_stop(self,value):
        if len(self.window)==0:
            self.__add_value(value)
            return False
        elif min(self.window)-value<self.threshold:
            self.patience-=1
            if not self.patience:
                print("Early Stopping.")
                return True
            else:
                return False
        else:
#            print("min is %s and val is %s "%(str(min(self.window)),str(value)))
            self.__add_value(value)
            self.patience=5
            return False

class PPrinter:

    def __init__(self):
        self.buffer=list()

    def add_to_buffer(self,_string):
        self.buffer.append(_string)

    def pprint(self):
        for _string in self.buffer:
            print(_string,end=" ")
        print()
        self.buffer=list()

class TrProgress:

    def __init__(self):
        self.loss=0
        self.estopper=EarlyStopper()
        self.printer=PPrinter()
        self.now=time.time()

    def check_progress(self,loss,epoch):
        self.loss+=loss
        if epoch%100==0 and epoch!=0:
            now=time.time()
            self.printer.add_to_buffer("Epoch time: %s"%(str(int(now-self.now))))
            self.now=now
            self.printer.add_to_buffer("Epoch: %s Loss: %10s"%(epoch,self.loss/100.0))
            self.printer.pprint()
            if self.estopper.early_stop(self.loss/100.0):
                return True
            else:
                self.loss=0
                return False
        else:
            return False



