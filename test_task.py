import numpy as np
import pandas as pd
import tensorflow as tf

learning_rate = 0.001
training_epochs = 100
batch_size = 5
display_step = 1


# read data
data = pd.read_csv('train.csv')
data = data.values
data = data.astype('float32')

# divide features and labels
features_data = data[:, :-1]
labels_data = data[:, -1]
labels_data = np.reshape(labels_data, (len(labels_data), 1))
print(features_data.shape, labels_data.shape)


# input parameters
n_features = features_data.shape[1]
n_units = data.shape[0]


# FIRST STEP (BUILDING GRAPH)
# make tensorflow graph
features = tf.placeholder(tf.float32, [None, n_features], name='features')
labels = tf.placeholder(tf.float32, [None, 1], name='labels')


# define weights for first layer
w1 = tf.Variable(tf.zeros([5, n_features]), name='weights1', trainable=True)

# first output
out1 = w1*features
print(out1.shape)

# define weights for the h2 layer
W2 = tf.Variable(tf.truncated_normal([int(out1.shape[1]), int(out1.shape[0])], mean=0, stddev=1 / np.sqrt(n_features)), name='weights2', trainable=True)
print(W2, out1.shape)
b2 = tf.Variable(tf.truncated_normal([int(out1.shape[0])], mean=0, stddev=1 / np.sqrt(n_features)), name='biases2')

# second output
out2 = tf.nn.tanh((tf.matmul(out1, W2)+b2), name='activationLayer2')

# define weights for the h3 layer
W3 = tf.Variable(tf.random_normal([int(out2.shape[1]), int(out2.shape[0])],mean=0,stddev=1/np.sqrt(n_features)), name='weights3', trainable=True)
b3 = tf.Variable(tf.zeros([int(out2.shape[0])]), name='biases3')

# third output
out3 = tf.nn.sigmoid((tf.matmul(out2, W3)+b3), name='activationLayer3')

# output layer weights and biasies
Wo = tf.Variable(tf.random_normal([int(out3.shape[1]), 1], mean=0, stddev=1/np.sqrt(n_features)), name='weightsOut', trainable=True)
bo = tf.Variable(tf.random_normal([1], mean=0, stddev=1/np.sqrt(n_features)), name='biasesOut')

# activation function(softmax)
a = tf.nn.softmax((tf.matmul(out3, Wo) + bo), name='activationOutputLayer')


# SECOND STEP (COST FUNCTION AND OPTIMIZER)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(cost)
# grads_and_vars = optimizer.compute_gradients(cost)
# # Change grads_and_vars as you wish
# opt_operation = optimizer.apply_gradients(grads_and_vars)

# THIRD STEP (TRAINING THE MODEL)
# creating a session

init = tf.global_variables_initializer()

#
# # 50 epochs with a smaller learning rate of 0.01
# from tensorflow.python.saved_model import tag_constants
# batch_size = 10
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(training_epochs):
#         for i in range(n_units):
#             _, c = sess.run([optimizer, cost], feed_dict={features: [features_data[i]],
#                                                           labels: [labels_data[0, i]]})
#
#         print(sess.run(w1))


# Launch the graph

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(features_data.shape[0]/batch_size)
        X_batches = np.array_split(features_data, total_batch)
        Y_batches = np.array_split(labels_data, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)

            _, c = sess.run([optimizer, cost], feed_dict={features: batch_x,
                                                          labels: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            print(sess.run(w1))
    print("Optimization Finished!")

