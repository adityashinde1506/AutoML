#!/usr/bin/env python3

class EarlyStopper:

    def __init__(self,patience=5,threshold=0.001):
        self.patience=patience
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
            print(min(self.window)-value)
            print("Patience decreased")
            self.patience-=1
            if not self.patience:

                return True
            else:
                return False
        else:
            print("min is %s and val is %s "%(str(min(self.window)),str(value)))
            self.__add_value(value)
            self.patience=5
            return False
