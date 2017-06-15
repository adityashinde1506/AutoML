from .helpers import EarlyStopper

class DataPrep:
    
    """
        Split dataframe into training data and labels.
        target_attr is the column name of target column.
    """

    def __init__(self,target_attr):
        self.target=target_attr

    def run(self,input_data):
        target=input_data[self.target].as_matrix()
        data=input_data.drop(self.target,axis=1).as_matrix()
        return data,target

class Stopper:

    """
        mode:
        1 - early stopping
        0 - epochs
    """
    mode=0

    def __init__(self,epochs):
        if epochs!=0:
            self.mode=0    
            self.epochs=epochs
        elif epochs==0:
            self.mode=1
            self.count=0
            self.e_stopper=EarlyStopper()

    def run(self,loss=None):
        if not self.mode:
            self.epochs-=1
            print("Epochs left: %d"%self.epochs)
            return self.epochs==0
        else:
            assert loss!=None
            self.count+=1
            print("Loss: %f after epoch: %d"%(loss,self.count))
            return self.e_stopper.early_stop(loss)
