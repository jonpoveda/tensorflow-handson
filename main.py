# -*- coding: utf-8 -*-
""" Trains and assess a neural network """

import utils_mnist
from models.lenet2 import LeNet
# from models.simple import SimpleModel
# from models.simple0 import Simple0
import tensorflow as tf
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def main():
    print(get_available_gpus())

    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,
                                            allow_soft_placement=True))
    # Runs the op.
    print(sess.run(c))

    dataset = utils_mnist.load_mnist_32x32()

    # model = Simple0()       # Simple model without wrappers
    # model = SimpleModel()   # Simple model with Estimators
    model = LeNet()         # LeNet model with wrappers

    model.train_and_evaluate(data=dataset,
                             batch_size=550,
                             num_epochs=100,
                             steps=None)


if __name__ == '__main__':
    main()
