from .experiments import *
import logging

logger=logging.getLogger(__name__)

class MetaExperiment(object):

    def __init__(self,experiments):
        logger.info("Starting experiments.")
        self.results=list()
        for name,experiment in experiments:
            logger.debug("Running experiment NAME:{} DESC:{}".format(name,str(experiment)))
            scores,mean=experiment.run_experiment()
            self.results.append((name,scores,mean))
        self.generate_report()

    def generate_report(self):
        for name,_,mean in self.results:
            logger.info("Experiment {} mean metric {}".format(name,mean))
