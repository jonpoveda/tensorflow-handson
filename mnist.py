import numpy as np
import tensorflow as tf

# def model(features, labels, mode):
# N samples of D values and H neurons per hidden layer
# N = 64  # samples
D = 32 * 32  # sample size
C = 10  # classes
H = 256  # units in hidden layers

input_data = tf.placeholder(tf.float32, shape=(None, D))
input_labels = tf.placeholder(tf.float32, shape=(None, C))

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
                         units=10,
                         kernel_initializer=init)

loss = tf.losses.mean_squared_error(logits, input_labels)
# classification_loss = \
# tf.keras.losses.categorical_crossentropy(predictions, input_labels)
predictions = {
    "classes": tf.argmax(input=logits, axis=1, name='classes'),
    "probabilities": tf.nn.softmax(logits, name='softmax_tensor')
}
onehot_labels = tf.one_hot(indices=tf.cast(input_labels, tf.int32), depth=10)
class_loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)
accuracy, update_op = tf.metrics.accuracy(labels=input_labels,
                                          predictions=predictions['classes'],
                                          name='accuracy')
my_acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(input_labels, tf.int64),
                                         predictions['classes']), tf.float32))

eval_metric_ops = {
    'accuracy': (accuracy, update_op)
}
acc = tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT,
                                 loss=class_loss,
                                 eval_metric_ops=eval_metric_ops)

# acc = tf.estimator.EstimatorSpec(predictions)
# acc, acc_op = tf.metrics.accuracy(predictions=tf.argmax(logits, 1),
#                                   labels=tf.argmax(input_labels, 1))

# learning_rate = 1e-5
# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer()
updates = optimizer.minimize(loss)

import utils_mnist
with tf.Session() as sess:
    mnist = utils_mnist.load_mnist_32x32()

    sess.run(tf.global_variables_initializer())


    def train(data: np.ndarray,
              labels: np.ndarray,
              n_samples: int = -1,
              batch_size: int = 6400,
              epochs: int = 100):
        """ Train the model

        Args:
            data: input data (images)
            labels: input labels
            n_samples: how many samples use to run a test, '-1' means all
                available
            batch_size: number of samples per batch
            epochs: number of epochs to run
        """
        n_batches = n_samples // batch_size
        if n_samples == -1:
            n_batches = data.shape[0] // batch_size

        loss_function = []

        for epoch in range(epochs):
            for batch_num in range(n_batches):
                batch_data: np.ndarray = data[
                                         batch_size * batch_num:
                                         batch_size * (batch_num + 1)]
                batch_labels = labels[
                               batch_size * batch_num:
                               batch_size * (batch_num + 1)]

                # Reshape input data to 1D-vector
                batch_data = np.reshape(batch_data, (batch_size, 32 * 32))

                # Build a one-hot vector from labels
                # batch_labels = tf.one_hot(batch_labels, depth=10).eval()

                values = {input_data: batch_data,
                          input_labels: batch_labels}

                loss_val = sess.run([loss, updates], feed_dict=values)
                loss_function.append(loss_val[0])

            print(f'Epoch: {epoch}, Loss: {loss_val[0]}')

        return loss_function


    def test(data: np.ndarray,
             labels: np.ndarray,
             n_samples: int = -1):
        """ Test the model

        Args:
            data: input data (images)
            labels: input labels
            n_samples: how many samples use to run a test, '-1' means
                use all available
        """
        batch_size = 1

        n_batches = n_samples // batch_size
        if n_samples == -1:
            n_batches = data.shape[0] // batch_size

        losses = []
        class_losses = []
        for batch_num in range(n_batches):
            batch_data: np.ndarray = data[batch_size * batch_num:
                                          batch_size * (batch_num + 1)]

            batch_labels = labels[batch_size * batch_num:
                                  batch_size * (batch_num + 1)]

            # Reshape input data to 1D-vector
            batch_data = np.reshape(batch_data, (batch_size, 32 * 32))

            # Build a one-hot vector from labels
            # batch_labels = tf.one_hot(batch_labels, depth=10).eval()

            values = {input_data: batch_data,
                      input_labels: batch_labels}

            # loss_val = sess.run([acc, ], feed_dict=values)
            loss_val = sess.run([acc, ], feed_dict=values)

            losses.append(loss_val[0])
            # class_losses.append(classification_loss[0])
            class_losses.append(acc)

        print(f'Loss: {loss_val[0]}, Class-loss: {acc}')
        class_losses = -1
        return losses, class_losses


    losses = train(data=mnist.train.images,
                   labels=mnist.train.labels,
                   n_samples=1000,
                   batch_size=500,
                   epochs=2)
    plt.figure('Training loss')
    plt.plot(losses)
    plt.savefig('task2_train_loss.png')

    losses, class_losses = test(data=mnist.test.images,
                                labels=mnist.test.labels,
                                n_samples=50)

    print(f'Error in test loss: {np.sum(losses)}')
    print(f'Error in test classif.: {np.sum(class_losses)}')
    plt.figure('Test loss')
    plt.plot(losses)
    plt.savefig('task2_test_loss.png')

    plt.figure('Test classification loss')
    plt.plot(losses)
    plt.savefig('task2_test_class_loss.png')

    plt.show()
