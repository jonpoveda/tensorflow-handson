# -*- coding: utf-8 -*-
""" Trains and assess a neural network """

import utils_mnist
from models.lenet import LeNet
from models.simple import SimpleModel
from models.simple0 import Simple0


def main():
    dataset = utils_mnist.load_mnist_32x32()

    model = Simple0()       # Simple model without wrappers
    # model = SimpleModel()   # Simple model with Estimators
    # model = LeNet()         # LeNet model with wrappers

    model.train_and_evaluate(data=dataset,
                             batch_size=550,
                             num_epochs=100,
                             steps=None)


if __name__ == '__main__':
    main()
