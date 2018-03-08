""" Implementation following TF doc """

import utils_mnist
from models.simple import SimpleModel
from models.lenet import LeNet

if __name__ == '__main__':
    dataset = utils_mnist.load_mnist_32x32()
    model = SimpleModel()
    # model = LeNet()

    eval_results = model.train_and_evaluate(data=dataset,
                                            batch_size=550,
                                            num_epochs=4,
                                            steps=None)
    print(eval_results)
