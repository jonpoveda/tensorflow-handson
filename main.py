""" Implementation following TF doc """

import utils_mnist
from models.basic_model import BasicModel

if __name__ == '__main__':
    dataset = utils_mnist.load_mnist_32x32()
    model = BasicModel()

    eval_results = model.train_and_evaluate(data=dataset,
                                            batch_size=550,
                                            num_epochs=2,
                                            steps=None)
    print(eval_results)
