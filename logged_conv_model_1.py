from __future__ import print_function
from __future__ import division
import numpy as np
import csv
import helpers
import subprocess

from constants import *
from nn_layers import *

import tensorflow as tf

TRAIN_DIR = "logs/train"
TEST_DIR = "logs/test"

def main():
    X, Y = helpers.pluck_data('data/train')
    X_test, Y_test = helpers.pluck_data('data/test')
    Y = helpers.one_hot_encode(Y)
    Y_test = helpers.one_hot_encode(Y_test)
    
    # Input
    x = tf.placeholder(tf.float32, [None, 3072]) 
    y = tf.placeholder(tf.float32, [None, 2]) # benign or mal => 2 classe
    
    # Convolution & Max Pool 1
    x_image = tf.reshape(x, [-1,32,32,3])
    
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # Convolution & Max Pool 2
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    conv = conv2d(h_pool1, W_conv2)
    conv_with_bias = conv + b_conv2
    h_conv2 = tf.nn.relu(conv_with_bias)
    
    h_pool2 = max_pool_2x2(h_conv2)
    
    # Layer 1
    W1 = weight_variable([8 * 8 * 64, 1024])
    bias_1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
    output_layer_1 = tf.nn.relu(tf.matmul(h_pool2_flat, W1) + bias_1)
    
    # Dropout 
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(output_layer_1, keep_prob)
    
    # Readout 
    W_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    # Softmax layer & Optimization
    pred = tf.nn.softmax(y_conv)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y))
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
    
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(TRAIN_DIR, sess.graph)
        sess.run(init)
        for epoch in range(training_epochs):
            X, Y = helpers.reshuffle_data(X,Y)
            avg_cost = 0.
            total_batch = int(X.shape[0]/batch_size)
            for i in range(total_batch): # start at i=1
                batch_xs, batch_ys = helpers.get_next_batch(i, X, Y, batch_size)
                if i == total_batch - 1:
                    train_accuracy = accuracy.eval(feed_dict={
                        x:batch_xs, 
                        y: batch_ys, 
                        keep_prob: 1.0}) # 1.0 keep probability to not drop out
                    print("step %d, training accuracy %g" % (i, train_accuracy))
    
                summary, _, c,prediction = sess.run([merged, optimizer, cost, pred], 
                        feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
                train_writer.add_summary(summary, i + epoch * total_batch)
                avg_cost += c / total_batch
    
            if (epoch+1) % display_step == 0:
                print("Epoch " + str(epoch) + " cost: " + str(avg_cost))
    
        output_file = open("tf_log_out.csv", 'a+')
        writer = csv.writer(output_file)
        total_batch = int(X_test.shape[0] / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = helpers.get_next_batch(i, X, Y, batch_size)
            _, c,prediction = sess.run([optimizer, cost,pred], 
                    feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            for i in range(prediction.shape[0]):
                writer.writerow([prediction[i].tolist().index(np.max(prediction[i])), 
                    batch_ys[i].tolist().index(1.0)])
            avg_cost += c / total_batch
        output_file.close()

def clean_logs():
    clean_dir(TRAIN_DIR)

def clean_dir(dr):
    try:
        subprocess.call('rm tf_log_out.csv')
    except OSError:
        print("No existing log output to remove.")
    if tf.gfile.Exists(dr):
        tf.gfile.DeleteRecursively(dr)
    tf.gfile.MakeDirs(dr)

if __name__ == "__main__":
    clean_logs()
    main()
