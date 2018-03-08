import tensorflow as tf

from models.base_model import BaseModel


# dropout= 0.5
# with tf.variable_scope(scope, 'LeNet', [images]):
#     net = end_points['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
#     net = end_points['pool1'] = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
#     net = end_points['conv2'] = slim.conv2d(net, 64, [5, 5], scope='conv2')
#     net = end_points['pool2'] = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
#     net = slim.flatten(net)
#     end_points['Flatten'] = net
#
#     net = end_points['fc3'] = slim.fully_connected(net, 1024, scope='fc3')
#     if not num_classes:
#         return net, end_points
#     net = end_points['dropout3'] = slim.dropout(
#         net, dropout_keep_prob, is_training=is_training, scope='dropout3')
#     logits = end_points['Logits'] = slim.fully_connected(
#         net, num_classes, activation_fn=None, scope='fc4')

class LeNet(BaseModel):
    def __init__(self):
        super(LeNet, self).__init__()

    def _model_function(self, features, labels, mode):
        """ Model function """
        # N samples of D values and H neurons per hidden layer
        width, height, channels = 32, 32, 1  # sample size
        n_classes = 10  # classes

        # Input layer
        input_layer = tf.reshape(features['x'], [-1, width, height, 1])

        # Dense layers
        init = tf.contrib.layers.xavier_initializer()
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=32, kernel_size=[5, 5],
                                 name='conv1')
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[2, 2], strides=2,
                                        name='pool1')
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64, kernel_size=[5, 5],
                                 name='conv2')
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[2, 2], strides=2,
                                        name='pool2')
        flat = tf.layers.flatten(pool2)
        fc3 = tf.layers.dense(inputs=flat,
                              units=1024,
                              activation=tf.nn.relu,
                              kernel_initializer=init,
                              name='fc3')
        dropout3 = tf.layers.dropout(inputs=fc3,
                                     rate=0.5,
                                     training=bool(
                                         mode == tf.estimator.ModeKeys.TRAIN),
                                     name='dropout3')
        logits = tf.layers.dense(inputs=dropout3,
                                 units=10,
                                 name='fc4')

        predictions = {
            'classes': tf.argmax(input=logits, axis=1, name='classes'),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        # Calculate loss
        one_hot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32),
                                    depth=n_classes)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels,
                                               logits=logits)
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
        #                                        logits=logits)

        # Config training
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss=loss,
                                          global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)

        # Add evaluation metrics
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels,
                                            predictions=predictions['classes'])
        }
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    def train_and_evaluate(self, data,
                           batch_size=100, num_epochs=None, steps=None):
        """ Trains and evaluates the model

        Args:
            data: mnist object
            batch_size: number of images to do one gradient update
            num_epochs: number of epochs to run, if `None` a number of steps are
                computed to run the equivalent to one epoch
            steps: number of steps to run,
        """
        if (num_epochs is None and steps is None) or \
            (num_epochs is not None and steps is not None):
            raise ValueError('Please set one of num_epochs or steps. '
                             'They are mutually exclusive')

        if num_epochs is None:
            steps = data.train.images.shape[0] // batch_size + 1

        tf.logging.set_verbosity(tf.logging.INFO)

        # Create estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=self._model_function,
            model_dir=f'tmp/mnist_{self.__class__.__name__}')

        # Train the model
        if num_epochs is None:
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': data.train.images},
                y=data.train.labels,
                batch_size=batch_size,
                shuffle=True)

            mnist_classifier.train(input_fn=train_input_fn,
                                   steps=steps)

        else:
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'x': data.train.images},
                y=data.train.labels,
                batch_size=batch_size,
                num_epochs=num_epochs,
                shuffle=True)

            mnist_classifier.train(input_fn=train_input_fn)

        # Evaluate model
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': data.test.images},
            y=data.test.labels,
            num_epochs=1,
            shuffle=False)

        return mnist_classifier.evaluate(input_fn=eval_input_fn)
