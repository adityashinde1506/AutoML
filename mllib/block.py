from .lib.preprocess import *
from .lib.sources import *

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
