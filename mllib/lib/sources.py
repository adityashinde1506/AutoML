import pandas

class CSVDataSource:

    def __init__(self):
        pass

    def set_csv_source(self,files):
        self.sources=files

    def __pad_last(self,batch,data):
        if data.shape[0] < batch:
            return self.__pad_last(batch,pandas.concat([data,data],ignore_index=True))
        else:
            return data[:batch]

    def batch_generator(self,batch_size=10000):
        source=pandas.read_csv(self.sources[0],chunksize=batch_size)
        while 1:
            data=next(source)
            if data.shape[0]!=batch_size:
                self.sources.insert(0,self.sources.pop())
                source=pandas.read_csv(self.sources[0],chunksize=batch_size)
                print("Generator reinitialized.")
                yield self.__pad_last(batch_size,data)
            else:
                yield data
