import pandas
import shelve
from sklearn.preprocessing import LabelEncoder,StandardScaler

class DSetHandler:

    def __init__(self):
        pass

    def import_from_csv(self,csv_path,store_id=1):
        if store_id:
            dset=pandas.read_csv(csv_path).fillna(method="bfill").fillna(method="ffill")
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

    def __make_new_label_frame(self,dframe):
        frame=list()
        for i in range(len(dframe.columns)):
            if dframe[dframe.columns[i]].dtype=="object":
                frame.append(LabelEncoder())
            else:
                frame.append(None)
        return frame

    def __make_new_scaler_frame(self):
        return StandardScaler()

    def apply_label_frame(self,dataset,frame,fit=1):
        assert len(dataset.columns)==len(frame)
        for i in range(len(frame)):
            if frame[i]!=None:
                if fit:
                    dataset[dataset.columns[i]]=frame[i].fit_transform(dataset[dataset.columns[i]].apply(str))
                else:
                    dataset[dataset.columns[i]]=frame[i].transform(dataset[dataset.columns[i]].apply(str))
        return dataset

    def apply_scaler_frame(self,dataset,scaler_frame,fit=1):
        if fit:
            return pandas.DataFrame(scaler_frame.fit_transform(dataset),columns=dataset.columns)
        else:
            return pandas.DataFrame(scaler_frame.transform(dataset),columns=dataset.columns)

    def process_labels(self,dataframe):
        labels_frame=self.__make_new_label_frame(dataframe)
        dataframe=self.apply_label_frame(dataframe,labels_frame)
        return dataframe,labels_frame

    def rescale(self,dataframe):
        scaler=self.__make_new_scaler_frame()
        dframe=self.apply_scaler_frame(dataframe,scaler)
        return dframe,scaler

    def write_data(self,dframe,labels,scaler,output):
        dframe.to_csv(output+"/data.csv")
        metadata=shelve.open(output+"/metadata")
        metadata["columns"]=dframe.columns
        metadata["labels"]=labels
        metadata["scaler"]=scaler
        metadata["id"]=self.ids
        metadata.close()

    def apply_meta(self,dataframe,meta_path):
        ids,scaler_frame,labels_frame=self.read_meta_file(meta_path)
        dataframe=self.apply_label_frame(dataframe,labels_frame,0)
        dataframe=self.apply_scaler_frame(dataframe,scaler_frame,0)
        return dataframe

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

    def read_meta_file(self,meta_file):
        metadata=shelve.open(meta_file)
        labels=metadata["labels"]
        scaler=metadata["scaler"]
        ids=metadata["id"]
        return ids,scaler,labels
