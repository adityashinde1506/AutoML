from scipy import stats
import numpy
import json

import logging

logger=logging.getLogger(__name__)

class Report(object):

    def __init__(self):
        self.experiments={}
        self.individuals=[]

    def add_result(self,name,trial_means,mean):
        self.experiments[name]={"trials":trial_means,"mean":mean}
        self.individuals.append(name)
        logger.debug("Added results for experiment {}".format(name))

    def run_tests(self):
        logger.debug("Running ttests for experiments.")
        ttest_results=numpy.zeros((len(self.individuals),len(self.individuals)))
        for i in range(len(self.individuals)):
            for j in range(len(self.individuals)):
                if i==j:
                    ttest_results[i][j]=numpy.nan
                else:
                    ttest_results[i][j]=stats.ttest_ind(
                                        self.experiments[self.individuals[i]]["trials"],
                                        self.experiments[self.individuals[j]]["trials"])[0]
        logger.info("ttests completed.")
        results=ttest_results.tolist()
        for i in range(len(results)):
            logger.info("Result {}:{}".format(i,json.dumps(results[i])))
        
