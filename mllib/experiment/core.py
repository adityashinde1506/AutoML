from .experiments import *
from .report import Report
import logging

logger=logging.getLogger(__name__)

class MetaExperiment(object):

    def __init__(self,experiments):
        self.report=Report()
        logger.info("Starting experiments.")
        self.results=list()
        for name,experiment in experiments:
            logger.debug("Running experiment NAME:{} DESC:{}".format(name,str(experiment)))
            scores,mean=experiment.run_experiment()
            self.results.append((name,scores,mean))
        self.generate_report()

    def generate_report(self):
        for name,trials,mean in self.results:
            self.report.add_result(name,trials,mean)
        self.report.run_tests()
