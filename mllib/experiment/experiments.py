import logging
import numpy


logger=logging.getLogger(__name__)

class Experiment(object):

    """
    Used for checking statiscal performance of a model on the testing data.
    args:
    model - a model according to sklearn specs.
    train - tuple of training dataset (X,y)
    test - tuple of testing dataset (X,y)
    metric - metric function to evaluate performance on.
    trials - number of trials.
    """

    def __init__(self,model,datasource,metric,trials=30,name="Unnamed experiment"):
        self.dataset=datasource
        self.source=str(datasource)
        del datasource
        self.model=model
        self.metric=metric
        self.trials=trials
        self.scores=list()
        self.name=name

    def single_run(self,train_X,train_y,test_X,test_y):
        self.model.fit(train_X,train_y)
        predictions=self.model.predict(test_X)
        return self.metric(test_y,predictions)

    def run_experiment(self):        
        dataset=self.dataset.get_dataset()
        train=dataset["train"]
        test=dataset["test"]
        train_X=train[0]
        train_y=train[1]
        test_X=test[0]
        test_y=test[1]
        logger.info("Starting experiment :{}.".format(self.name))
        for i in range(self.trials):
            score=self.single_run(train_X,train_y,test_X,test_y)
            logger.debug("Trial: {} metric: {}".format(i,score))
            self.scores.append(score)
        mean_score=numpy.array(self.scores).mean()
        logger.info("Finished experiment {}. Mean Metric for {} trials is {}".format(self.name,self.trials,mean_score))
        return numpy.array(self.scores),mean_score

    def __str__(self):
        desc="Experiment: NAME:{} MODEL:{} METRIC:{} TRIALS:{} DATA:{}".format(self.name,str(self.model),str(self.metric),self.trials,self.source)
        return desc




