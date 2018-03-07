import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# N samples of D values and H neurons per hidden layer
N, D, H = 64, 1000, 100

input_data = tf.placeholder(tf.float32, shape=(N, D))
labels = tf.placeholder(tf.float32, shape=(N, D))

init = tf.contrib.layers.xavier_initializer()
hidden = tf.layers.dense(inputs=input_data, units=H,
                         activation=tf.nn.relu, kernel_initializer=init)
hidden2 = tf.layers.dense(inputs=hidden, units=H,
                          activation=tf.nn.relu, kernel_initializer=init)
predicted_labels = tf.layers.dense(inputs=hidden, units=D,
                                   kernel_initializer=init)

loss = tf.losses.mean_squared_error(predicted_labels, labels)

learning_rate = 1e-5
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
updates = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {input_data: np.random.randn(N, D) + 10,
              labels: np.random.randn(N, D) * 2 + 10}

    losses = []
    for iteration in range(6000):
        loss_val = sess.run([loss, updates], feed_dict=values)
        losses.append(loss_val[0])

        if not iteration % 100:
            print(f'Iteration: {iteration}, Loss: {loss_val[0]}')
            last_loss = loss_val[0]

    plt.figure('Loss')
    plt.plot(losses)
    plt.savefig('task1_loss.png')
    plt.show()
