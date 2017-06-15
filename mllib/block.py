from .lib.preprocess import *
from .lib.sources import *
from .lib.training import *
from .models.model import ModelBuilder

class Block:

    def __init__(self):
        self.components=list()

    def show_components(self):
        print(self.components)

    def run(self,input_data,state):
        for component in self.components:
            component.set_state(state)
            input_data=component.run(input_data)
        return input_data

class PreprocessBlock(Block):

    operations={"standard":StandardScalerBlock,"min_max":MinMaxBlock}

    def __init__(self,components,args):
        super(PreprocessBlock,self).__init__()
        assert len(components)==len(args)
        for i in range(len(components)):
            self.components.append(self.operations[components[i]](**args[i]))

class SourceBlock:

    def run(self):
        return next(self.generator)

class CSVSourceBlock(SourceBlock):
    
    def __init__(self,files,batch_size):
        self.source=CSVDataSource()
        self.source.set_csv_source(files)
        self.generator=self.source.batch_generator(batch_size)

class ColumnFilter:

    def __init__(self,columns=[]):
        self.to_drop=columns

    def run(self,input_data):
        for column in self.to_drop:
            input_data=input_data.drop(column,axis=1)
        return input_data

class ModelBlock:

    def __init__(self,model,target_attr,epochs,data_source):
        self.model_builder=ModelBuilder()
        self.data_prep=DataPrep(target_attr)
        self.stopper=Stopper(epochs)
        self.data=data_source
        self.model=self.model_builder.get_model(**model)

    def run(self,state):
        if state=="train":
            while 1:
                data=self.data(state)
                X,y=self.data_prep.run(data)
                loss=self.model.train(X,y)
                if self.stopper.run(loss):
                    break



