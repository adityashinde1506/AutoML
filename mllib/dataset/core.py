import pandas
import fileinput
import logging

logger=logging.getLogger(__name__)

class Dataset(object):
    """
    Base object for any dataset.
    """

    def __init__(self,files,delimiter=",",chunk=None):
        logger.debug("Creating data iterators")
        if delimiter==" ":
            self.data_iterators=list(map(lambda x:pandas.read_csv(x,delim_whitespace=True,index_col=False,header=None,iterator=True,chunksize=None),files))
        else:
            self.data_iterators=list(map(lambda x:pandas.read_csv(x,delimiter=delimiter,index_col=False,header=None,iterator=True,chunksize=None),files))
        logger.debug("Iterators created.")
        logger.info("Dataset loaded.")

    def collect(self):
        logger.debug("Collecting all files into single dataframe.")
        return pandas.concat(list(map(lambda x:next(x),self.data_iterators)),ignore_index=True)
