from .block import *
import json

class Pipeline:

    block_type={"csv_source":CSVSourceBlock,"preprocessing":PreprocessBlock}

    def __init__(self,filename):
        self.blocks=list()
        self.specs=json.load(open(filename))
        self.__init_blocks(self.specs)

    def __init_block(self,arg_dict):
        block_type=arg_dict["type"]
        args=arg_dict["args"]
        self.blocks.append(self.block_type[block_type](**args))

    def __init_blocks(self,blocks):
        for block in blocks:
            self.__init_block(block["desc"])

    def train_run(self):
        data=self.blocks[0].run()
        for block in self.blocks[1:]:
            data=block.run(data,"train")
        return data

    def debug(self):
        print(self.blocks)
        for block in self.blocks[1:]:
            block.show_components()
