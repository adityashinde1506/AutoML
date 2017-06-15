from .sklearn_models.rootmodel import SKRootModel

class ModelBuilder:

    def __init__(self):
        pass

    def get_model(self,model_type,specs):
        if model_type=="sklearn":
            return SKRootModel(specs)
