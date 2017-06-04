import pandas
import shelve
from sklearn.preprocessing import LabelEncoder,StandardScaler

class DSetHandler:

    def __init__(self):
        pass

    def import_from_csv(self,csv_path,store_id=1):
        if store_id:
            dset=pandas.read_csv(csv_path).fillna(0)
            self.ids=dset["id"]
            return dset.drop("id",axis=1)
        else:
            self.ids=None
            return pandas.read_csv(csv_path).fillna(0)

    def seperate_target(self,dataframe,label_column=None):
        if label_column==None:
            label_column=dataframe.columns[-1]
        labels=dataframe[label_column]
        dataset=dataframe.drop(label_column,axis=1)
        return dataset,labels

    def __get_column_dtypes(self,dataframe):
        return list(map(lambda x:dataframe[x].dtype,dataframe.columns))

    def process_labels(self,dataframe):
        dtypes=self.__get_column_dtypes(dataframe)
        label_transform=list()
        for i in range(len(dtypes)):
            if dtypes[i]=="object":
                ltransform=LabelEncoder()
                dataframe[dataframe.columns[i]]=ltransform.fit_transform(dataframe[dataframe.columns[i]].apply(str))
                label_transform.append(ltransform)
            else:
                label_transform.append(None)
        return dataframe,label_transform

    def rescale(self,dframe):
        scaler=StandardScaler()
        dframe=pandas.DataFrame(scaler.fit_transform(dframe),columns=dframe.columns)
        return dframe,scaler

    def write_data(self,dframe,labels,scaler,output):
        dframe.to_csv(output+"/data.csv")
        metadata=shelve.open(output+"/metadata")
        metadata["columns"]=dframe.columns
        metadata["labels"]=labels
        metadata["scaler"]=scaler
        metadata["id"]=self.ids
        metadata.close()

    def preprocess_with_labels(self,dataframe,output_path):
        dframe,self.ltransform=self.process_labels(dataframe)
        dframe,self.scaler=self.rescale(dframe)
        self.write_data(dframe,self.ltransform,self.scaler,output_path)

    def preprocess_wo_labels(self,dataframe,output_path):
        dframe,self.scaler=self.rescale(dataframe)
        self.write_data(dframe,None,self.scaler,output_path)

    def get_training_set(self,csv_path):
        dframe=self.import_from_csv(csv_path,0)
        dframe=dframe.drop(dframe.columns[0],axis=1)
        X,target=self.seperate_target(dframe)
        return X.as_matrix(),target.as_matrix()

    def get_testing_set(self,csv_path): 
        dframe=self.import_from_csv(csv_path,0)
        dframe=dframe.drop(dframe.columns[0],axis=1)
        return dframe.as_matrix()



