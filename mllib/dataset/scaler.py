import logging

logger=logging.getLogger(__name__)

class Scaler(object):

    """
    A simple wrapper around scikit's scalers.
    args:
        scaler- a reference to a scikit scaler.
    """

    def __init__(self,scaler):
        self.scaler_data=scaler()
        logger.debug("Data scaler initialized: {}.".format(str(self.scaler_data)))
        self.scaler_target=scaler()
        logger.debug("Target scaler initialized: {}.".format(str(self.scaler_target)))

    def scale_train_data(self,X):
        logger.debug("Scaling training data.")
        return self.scaler_data.fit_transform(X)

    def scale_test_data(self,X):
        logger.debug("Scaling test data.")
        return self.scaler_data.transform(X)

    def scale_train_target(self,y):
        logger.debug("Scaling training target.")
        return self.scaler_target.fit_transform(y.reshape(-1,1)).reshape(-1)

    def scale_test_target(self,y):
        logger.debug("Scaling test target.")
        return self.scaler_target.transform(y.reshape(-1,1)).reshape(-1)

    def rescale_data(self,X):
        logger.debug("Rescaling data.")
        return self.scaler_data.inverse_transform(X)

    def rescale_target(self,y):
        logger.debug("Rescaling target.")
        return self.scaler_target.inverse_transform(y.reshape(-1,1)).reshape(-1)
