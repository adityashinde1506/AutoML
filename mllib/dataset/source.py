import logging
import pandas

from .core import Dataset
from sklearn.model_selection import train_test_split

logger=logging.getLogger(__name__)

class Datasource(object):

    def __init__(self,files,headers,delimiter=",",chunk=None,remove_cols=[],remove_ind=[],transforms=[],test_split=0.3,target_col="y"):
        self.dataset=Dataset(files,delimiter=delimiter,chunk=chunk)
        self.remove_cols=remove_cols
        self.remove_ind=remove_ind
        self.headers=headers
        self.transforms=transforms
        if chunk==None:
            self.IN_MEM=True
        else:
            self.IN_MEM=False
        self.target=target_col
        self.split=test_split
        logger.info("Datasource initialized.")
        logger.debug("Datasource: MEM:{} DISCARD:{} TARGET:{}".format(self.IN_MEM,",".join(self.remove_cols),self.target))

    def __remove_cols(self,dataframe):
        for col in self.remove_cols:
            logger.debug("Dropping column {}".format(col))
            dataframe=dataframe.drop(col,axis=1)
        # TODO: drop based on indexing.
        return dataframe

    def __get_data_matrices(self,dataframe):
        y=dataframe[self.target].as_matrix()
        X=dataframe.drop(self.target,axis=1).as_matrix()
        return X,y

    def __apply_transforms(self,frame):
        for transform in self.transforms:
            logger.debug("Applying transform {}".format(str(transform)))
            frame=transform(frame)
        return frame

    def prepare_sets(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=self.split,random_state=42)
        self.sets={"train":(X_train,y_train),"test":(X_test,y_test)}
        logger.debug("Datasets prepared.")

    def get_dataset(self):
        if self.IN_MEM:
            logger.debug("Getting full dataset.")
            data=self.dataset.collect()
            data.columns=self.headers
            data=self.__apply_transforms(data)
            data=self.__remove_cols(data)
            X,y=self.__get_data_matrices(data)
            self.prepare_sets(X,y)
            return self.sets
        else:
            # TODO: return batches of data.
            return None

    def get_dataset_asframe(self):
        if self.IN_MEM:
            logger.debug("Getting dataframe")
            data=self.dataset.collect()
            data.columns=self.headers
            data=self.__apply_transforms(data)
            data=self.__remove_cols(data)
            return data
        else:
            # TODO: return batch
            return None
