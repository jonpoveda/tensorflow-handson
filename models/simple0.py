# -*- coding: utf-8 -*-
""" Implementation of a simple neural network without TF wrappers """
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from models.base_model import BaseModel


class Simple0(BaseModel):
    def __init__(self):
        super(Simple0, self).__init__()

    def _model_function(self, features, labels, mode):
        raise NotImplementedError

    def train_and_evaluate(self, data, batch_size, num_epochs, steps=None):
        # Define graph
        # N samples of D values and H neurons per hidden layer
        # N = 64  # samples
        D = 32 * 32  # sample size
        C = 10  # classes
        H = 256  # units in hidden layers

        input_data = tf.placeholder(tf.float32, shape=(None, D))
        input_labels = tf.placeholder(tf.int32, shape=(None,))

        input_labels_one_hot = tf.one_hot(
            indices=tf.cast(input_labels, tf.int32), depth=C, dtype=tf.int32)

        init = tf.contrib.layers.xavier_initializer()

        hidden = tf.layers.dense(inputs=input_data,
                                 units=H,
                                 activation=tf.nn.relu,
                                 kernel_initializer=init)

        hidden2 = tf.layers.dense(inputs=hidden,
                                  units=H,
                                  activation=tf.nn.relu,
                                  kernel_initializer=init)

        logits = tf.layers.dense(inputs=hidden2,
                                 units=C,
                                 kernel_initializer=init)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=input_labels_one_hot,
            logits=logits)

        # Update on training
        update_operation = tf.train.AdamOptimizer().minimize(loss)

        predictions = {
            "classes": tf.argmax(input=logits, axis=1, name='classes'),
            "probabilities": tf.nn.softmax(logits, name='softmax_tensor')
        }

        accuracy, accuracy_update_op = tf.metrics.accuracy(
            labels=input_labels,
            predictions=predictions['classes'],
            name='accuracy')

        # Update running vars behind the scene for accuracy
        running_vars = \
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='accuracy')
        running_vars_initializer = \
            tf.variables_initializer(var_list=running_vars)

        with tf.Session() as sess:
            mnist = data

            sess.run([tf.global_variables_initializer(),
                      running_vars_initializer])

            def train(samples: np.ndarray, labels: np.ndarray,
                      n_samples: int = -1,
                      batch_size: int = 6400, epochs: int = 100):
                """ Train the model

                Args:
                    samples: input data (images)
                    labels: input labels
                    n_samples: how many samples use to run a test, '-1' means all
                        available
                    batch_size: number of samples per batch
                    epochs: number of epochs to run
                """
                n_batches = n_samples // batch_size
                if n_samples == -1:
                    n_batches = samples.shape[0] // batch_size

                loss_function = []
                for epoch in range(epochs):
                    for batch_num in range(n_batches):
                        batch_data: np.ndarray = \
                            samples[batch_size * batch_num:
                                    batch_size * (batch_num + 1)]
                        batch_labels: np.ndarray = \
                            labels[batch_size * batch_num:
                                   batch_size * (batch_num + 1)]

                        # Reshape input data to 1D-vector
                        batch_data = np.reshape(batch_data,
                                                (batch_size, 32 * 32))

                        values = {input_data: batch_data,
                                  input_labels: batch_labels}

                        metrics = sess.run([update_operation, loss, ],
                                           feed_dict=values)
                        loss_function.append(metrics[1])

                    print(f'Epoch: {epoch}, loss: {metrics[1]}')

                return loss_function

            def test(samples: np.ndarray,
                     labels: np.ndarray,
                     batch_size: int = 5500,
                     n_samples: int = -1):
                """ Test the model

                Args:
                    samples: input data (images)
                    labels: input labels
                    n_samples: how many samples use to run a test, '-1' means
                        use all available
                    batch_size: test in batches (does not affect on the global
                        metrics)
                """
                n_batches = n_samples // batch_size
                if n_samples == -1:
                    n_batches = samples.shape[0] // batch_size

                for batch_num in range(n_batches):
                    batch_data: np.ndarray = samples[batch_num * batch_size:
                                                     (
                                                         batch_num + 1) * batch_size]

                    batch_labels = labels[batch_num * batch_size:
                                          (batch_num + 1) * batch_size]

                    # Reshape input data to 1D-vector
                    batch_data = np.reshape(batch_data,
                                            (batch_size, 32 * 32))

                    values = {input_data: batch_data,
                              input_labels: batch_labels}

                    metrics = sess.run([accuracy_update_op, accuracy],
                                       feed_dict=values)

                print(f'Final accuracy: {metrics[1]}')
                return metrics[1]

            train_labels = mnist.train.labels
            loss_function = train(samples=mnist.train.images,
                                  labels=train_labels,
                                  n_samples=batch_size * 3,
                                  batch_size=batch_size,
                                  epochs=num_epochs)
            plt.figure('Training loss')
            plt.plot(loss_function)
            plt.savefig('task2_train_loss.png')

            test_labels = mnist.test.labels
            acc = test(samples=mnist.test.images,
                       labels=test_labels,
                       batch_size=batch_size,
                       n_samples=-1)

            print(f'Accuracy in test: {acc}')
