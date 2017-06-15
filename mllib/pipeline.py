from .block import *
import json

class Pipeline:

    block_type={"csv_source":CSVSourceBlock}

    def __init__(self,filename):
        self.specs=json.load(open(filename))
        self.__init_source(self.specs["source"])
        self.__init_filter(self.specs["filter"])
        self.__init_preprocess(self.specs["preprocess"])
        self.__init_model(self.specs["model"])
    
    def __init_source(self,config):
        self.source=self.block_type[config["type"]](**config["args"])

    def __init_filter(self,config):
        self.filter=ColumnFilter(config["columns"])

    def __init_preprocess(self,config):
        self.preprocess=PreprocessBlock(**config)

    def __init_model(self,config):
        self.model=ModelBlock(**config,data_source=self.run_data_section)

    def run_data_section(self,state="train"):
        data=self.source.run()
        data=self.filter.run(data)
        data=self.preprocess.run(data,state)
        return data

    def train(self):
        self.model.run("train")
