import pandas

class DSetHandler:

    def __init__(self):
        pass

    def import_from_csv(self,csv_path):
        return pandas.read_csv(csv_path)

    def seperate_labels(self,dataframe,label_column=None):
        if label_column==None:
            label_column=dataframe.columns[-1]
        labels=dataframe[label_column]
        dataset=dataframe.drop(label_column,axis=1)
        return dataset,labels
