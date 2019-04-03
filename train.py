import numpy as np
import tensorflow as tf
import os
import csv

def multilayer_perceptron(X, weight, bias):
    layer1 = tf.nn.relu(tf.add(tf.matmul(X, weight['h1']), bias['h1']))
    out_layer = tf.nn.relu(tf.add(tf.matmul(layer1, weight['output']), bias['output']))

    return out_layer

# Fetch dataset (code from experiments.py)
dataset = np.load('./datasets/v0_eigens.npz')

train_data_size = dataset['train_eigens'].shape[0]
valid_data_size = train_data_size // 5
train_data_size = train_data_size - valid_data_size

train_data = dataset['train_eigens'][:train_data_size]
valid_data = dataset['train_eigens'][train_data_size:]

train_eigens = train_data[:, :-28].reshape(-1, 896)
train_labels = train_data[:, -28:]

valid_eigens = valid_data[:, :-28].reshape(-1, 896)
valid_labels = valid_data[:, -28:]

test_eigens = dataset['issue_eigens'][:, :-28].reshape(-1, 896)

# Parameters
learning_rate = 0.001
training_epoch = 100
batch_size = 50
display_step = 1

n_input = 896
n_hidden = 462
n_output = 28

weight = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'output': tf.Variable(tf.random_normal([n_hidden, n_output]))
}

bias = {
    'h1': tf.Variable(tf.random_normal([n_hidden])),
    'output': tf.Variable(tf.random_normal([n_output]))
}

X = tf.placeholder('float', [None, n_input])
Y = tf.placeholder('float', [None, n_output])

# Initialization
pred = multilayer_perceptron(X, weight, bias)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    
    # Training
    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(train_data_size / batch_size)

        for i in range(total_batch):
            _, c = sess.run([optimizer, cost], feed_dict={X: train_eigens[i*batch_size : (i+1)*batch_size, :],
                                                          Y: train_labels[i*batch_size : (i+1)*batch_size, :]})
            
            avg_cost += c / total_batch

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    acc = accuracy.eval({X: valid_eigens, Y: valid_labels})
    print('Accuracy: ', acc)