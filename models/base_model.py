class BaseModel(object):
    def __init__(self):
        pass

    def model_function(self, features, labels, mode):
        pass

    def train_and_evaluate(self, data, batch_size=100,
                           num_epochs=None, steps=None):
        pass
