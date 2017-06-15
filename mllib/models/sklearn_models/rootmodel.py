import importlib

class SKRootModel:

    def __init__(self,specs):
        desc=specs.split(".")
        modelname=desc[-1]
        module=importlib.import_module(".".join(desc[:-1]))
        self.model=getattr(module,modelname)(verbose=2)

    def train(self,X,y):
        self.model.partial_fit(X,y)
        return None
