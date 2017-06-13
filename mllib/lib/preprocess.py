from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas

class Preprocessor:

    def __init__(self,operation=None):
        self.operation=operation
        self.STATE="train"

    def run(self,input_data):
        columns=input_data.columns
        if self.STATE=="train":
            self.operation.partial_fit(input_data)
            input_data=self.operation.transform(input_data)
            return pandas.DataFrame(input_data,columns=columns)
        elif self.STATE=="test":
            input_data=self.operation.transform(input_data)
            return pandas.DataFrame(input_data,columns)
        else:
            raise Exception("Not Implemented!")

    def set_state(self,state="train"):
        self.STATE=state

class StandardScalerBlock(Preprocessor):

    def __init__(self):
        super(StandardScalerBlock,self).__init__(StandardScaler())

class MinMaxBlock(Preprocessor):

    def __init__(self,_min=0,_max=1):
        super(MinMaxBlock,self).__init__(MinMaxScaler(feature_range=(_min,_max)))
