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

        with tf.name_scope('input'):
            input_data = tf.placeholder(tf.float32, shape=(None, D))
            input_labels = tf.placeholder(tf.int32, shape=(None,))

            input_labels_one_hot = tf.one_hot(
                indices=tf.cast(input_labels, tf.int32), depth=C,
                dtype=tf.int32)

        with tf.name_scope('hidden'):
            init = tf.contrib.layers.xavier_initializer()

            hidden = tf.layers.dense(inputs=input_data,
                                     units=H,
                                     activation=tf.nn.relu,
                                     kernel_initializer=init)

            hidden2 = tf.layers.dense(inputs=hidden,
                                      units=H,
                                      activation=tf.nn.relu,
                                      kernel_initializer=init)

        with tf.name_scope('predictions'):
            logits = tf.layers.dense(inputs=hidden2,
                                     units=C,
                                     kernel_initializer=init)

        with tf.name_scope('train_metrics'):
            loss = tf.losses.softmax_cross_entropy(
                onehot_labels=input_labels_one_hot,
                logits=logits)

        # Update on training
        update_operation = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('test_metrics'):
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
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                              scope='test_metrics/accuracy')
        running_vars_initializer = \
            tf.variables_initializer(var_list=running_vars)

        # Log metrics
        summary_op_train = tf.summary.scalar('loss', loss)
        summary_op_test = tf.summary.scalar('accuracy', accuracy)
        summary_op = tf.summary.merge_all()

        # Logging writer
        writer = tf.summary.FileWriter(
            f'tmp/{self.__class__.__name__}/tensorboard',
            graph=tf.get_default_graph())

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
                i = 0
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

                        _, metric, summary = sess.run(
                            [update_operation, loss, summary_op_train],
                            feed_dict=values)
                        writer.add_summary(summary, i)
                        i += 1
                        loss_function.append(metric)

                    print(f'Epoch: {epoch}, loss: {metric}')

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

                i = 0
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

                    _, metric, summary = sess.run([accuracy_update_op,
                                                   accuracy,
                                                   summary_op_test],
                                                  feed_dict=values)
                    writer.add_summary(summary, i)
                    i += 1

                print(f'Final accuracy: {metric}')
                return metric

            train_labels = mnist.train.labels
            loss_function = train(samples=mnist.train.images,
                                  labels=train_labels,
                                  n_samples=-1,
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
