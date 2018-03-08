class BaseModel(object):
    def __init__(self):
        pass

    def _model_function(self, features, labels, mode):
        raise NotImplementedError(
            'This method has to be implemented in the subclass')

    def train_and_evaluate(self, data, batch_size, num_epochs, steps):
        raise NotImplementedError(
            'This method has to be implemented in the subclass')

