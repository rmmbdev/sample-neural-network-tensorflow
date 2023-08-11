import tensorflow as tf
import numpy as np
import utils
import math
from tqdm import tqdm

data = np.load('iris-data.npy')
train_features = data[:, 0:-1]
train_labels = data[:, -1]
train_labels = train_labels[:] - 1

has_train = not False

# model
class_count = 3
features_count = len(train_features[0, :])
one_hot_labels = utils.to_onehot(train_labels, class_count)

sess = tf.Session()

x = tf.placeholder("float", [None, features_count])
y_ = tf.placeholder("float", [None, class_count])

num_hidden = 16
W1 = tf.Variable(tf.truncated_normal([features_count, num_hidden], stddev=1. / math.sqrt(features_count)))
b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
h1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

num_hidden2 = 32
W2 = tf.Variable(tf.truncated_normal([num_hidden, num_hidden2], stddev=1. / math.sqrt(class_count)))
b2 = tf.Variable(tf.constant(0.1, shape=[num_hidden2]))
h2 = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([num_hidden2, class_count], stddev=1. / math.sqrt(class_count)))
b3 = tf.Variable(tf.constant(0.1, shape=[class_count]))

y = tf.nn.softmax(tf.matmul(h2, W3) + b3)

# train
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y + 1e-20, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())

if has_train:
    epoches = 10000
    train_acc = np.zeros(math.floor(epoches / 10))
    for i in tqdm(range(epoches)):
        if i % 10 == 0:
            A = accuracy.eval(session=sess,
                              feed_dict={x: train_features.reshape([-1, features_count]), y_: one_hot_labels})
            train_acc[i // 10] = A
        train_step.run(session=sess, feed_dict={x: train_features.reshape([-1, features_count]), y_: one_hot_labels})
print('Train accuracy over epoches: ', train_acc)

# calculate accuracy
# missclassified

# train
pred = np.argmax(y.eval(session=sess, feed_dict={x: train_features.reshape([-1, features_count])}), axis=1)
train_errors = []
for i in range(len(pred)):
    if pred[i] != train_labels[i]:
        train_errors.append(0)
    else:
        train_errors.append(1)
train_accuracy = np.sum(train_errors) / len(pred)
print('Train Accuracy is: ', train_accuracy)
