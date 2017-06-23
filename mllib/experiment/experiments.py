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

    def __init__(self,model,datasource,metric,trials=30):
        dataset=datasource.get_dataset()
        self.source=str(datasource)
        del datasource
        train=dataset["train"]
        test=dataset["test"]
        self.model=model
        self.train_X=train[0]
        self.train_y=train[1]
        self.test_X=test[0]
        self.test_y=test[1]
        self.metric=metric
        self.trials=trials
        self.scores=list()

    def single_run(self):
        self.model.fit(self.train_X,self.train_y)
        predictions=self.model.predict(self.test_X)
        return self.metric(self.test_y,predictions)

    def run_experiment(self):
        logger.info("Starting experiment.")
        for i in range(self.trials):
            score=self.single_run()
            logger.debug("Trial: {} metric: {}".format(i,score))
            self.scores.append(score)
        mean_score=numpy.array(self.scores).mean()
        logger.info("Finished experiment. Mean Metric for {} trials is {}".format(self.trials,mean_score))
        return numpy.array(self.scores),mean_score

    def __str__(self):
        desc="Experiment: MODEL:{} METRIC:{} TRIALS:{} DATA:{}".format(str(self.model),str(self.metric),self.trials,self.source)
        return desc




