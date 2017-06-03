import pandas
import shelve


class Reconstructor:

    def reconstruct(self,input_file,shelve_file):
        dataset=pandas.read_csv(input_file)
        metadata=shelve.open(shelve_file)
        for column in dataset.columns:
            if column not in metadata["columns"]:
                dataset=dataset.drop(column,axis=1)
            dataset=reconstruct_scale(dataset,metadata["scaler"])
        dataset=reconstruct_labels(dataset,metadata["labels"])
        metadata.close()
        print(dataset[:5])

    def reconstruct_scale(self,dataset,scaler):
        cols=dataset.columns
        return pandas.DataFrame(scaler.inverse_transform(dataset),columns=cols)

    def reconstruct_labels(self,dataset,labels):
        for i in range(len(labels)):
            encoder=labels[i]
            if encoder != None:
                dataset[dataset.columns[i]]=labels[i].inverse_transform(dataset[dataset.columns[i]].apply(int))
        return dataset
