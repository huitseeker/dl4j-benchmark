#!/usr/bin/env python

"""
TensorFlow Lenet

 Reference:
    https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py

TensorFlow install instructions: https://tensorflow.org/get_started/os_setup.html
MNIST tutorial: https://tensorflow.org/tutorials/mnist/tf/index.html
"""

import tensorflow as tf
import math
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
from six.moves import urllib, xrange
import numpy as np

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = mnist.IMAGE_PIXELS

FLAGS = tf.app.flags.FLAGS
# max_iteration = (epochs * numExamples)/batchSize (11 * 60000)/66
tf.app.flags.DEFINE_integer('max_iter', 10000, 'Number of iterations to run trainer.')
tf.app.flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
tf.app.flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
tf.app.flags.DEFINE_integer('batch_size', 66, 'Batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_integer('test_batch_size', 100, 'Test batch size. Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
tf.app.flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('l2', 1e-4, 'Weight decay.')

# TODO add gpu functionality


def load_data():
    return input_data.read_data_sets("/tmp/data/", one_hot=True)

def fill_feed_dict(data_set, images_pl, labels_pl):
    """Fills the feed_dict for training the given step.
    """
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    """
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def init_weights(shape):
    (fan_in, fan_out) = shape
    low = -1*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
    high = 1*np.sqrt(6.0/(fan_in + fan_out))
    weights = tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    weight_decay = tf.mul(tf.nn.l2_loss(weights), FLAGS.l2, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    return weights
    # return tf.Variable(tf.truncated_normal(shape,
    #                             stddev=1.0 / math.sqrt(float(shape[0]))), name='weights')

def inference(images, hidden1_units, hidden2_units):
    """Build the MNIST model up to where it may be used for inference.
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = init_weights([IMAGE_PIXELS, hidden1_units])
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable([hidden1_units, hidden2_units])
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable([hidden2_units, NUM_CLASSES])
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def score(logits, labels):
    """Calculates the loss from the logits and the labels.
    """
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,onehot_labels,name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def training(loss):
    """Sets up the training Ops.
    """
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)
    # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def run_training(train_data):
    """Train MNIST for a number of steps."""

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        logits = inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)

        # Add to the Graph the Ops for loss calculation.
        loss = score(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Add the variable initializer Op.
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.data_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for _ in xrange(FLAGS.max_iter):
            feed_dict = fill_feed_dict(train_data, images_placeholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

        return sess, logits, images_placeholder, labels_placeholder


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    """
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def do_eval(sess, logits, images_placeholder, labels_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    num_examples = data_set.num_examples
    steps_per_epoch = data_set.num_examples // FLAGS.test_batch_size
    for _ in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(evaluation(logits, labels_placeholder), feed_dict=feed_dict)
    precision = true_count / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def main():
    start_time = time.time()
    data_sets = load_data()
    sess, logits, images_placeholder, labels_placeholder = run_training(data_sets.train)
    do_eval(sess, logits, images_placeholder, labels_placeholder, data_sets.test)
    duration = time.time() - start_time
    print('Total train time: %s' % duration)


if __name__ == "__main__":
    main()