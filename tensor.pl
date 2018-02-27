import tensorflow as tf
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

import csv
import os
import random
from pprint import pprint
import matplotlib.pyplot as plt

DATA_DIR = '/home/pi/Downloads/'
TRAINING_FILE = 'RealEstate.csv'
saver = tf.train.Saver()

# Parameters
learning_rate = 0.01
training_epochs = 300
display_step = 25

def readcsv(csvfile):
    # [{'Bathrooms': '3',
    #'Bedrooms': '3',
    #'Location': 'Arroyo Grande',
    #'MLS': '132842',
    #'Price': '795000.00',
    #'Price/SQ.Ft': '335.30',
    #'Size': '2371',
    #'Status': 'Short Sale'}]
    with open(csvfile) as f:
        a = [{k: v for k, v in row.items()}
             for row in csv.DictReader(f, skipinitialspace=True)]
    return a

def extract_train(field, list_of_dicts):
    return [d[field] for d in list_of_dicts]

def run():
    training_file_path = os.path.join(DATA_DIR, TRAINING_FILE)
    training_data = readcsv(training_file_path)
    train_Y = numpy.asarray(extract_train('Price/SQ.Ft', training_data)).astype(numpy.float)
    train_X = numpy.asarray(extract_train('Size', training_data)).astype(numpy.float)
    n_samples = train_X.shape[0]
    rng = numpy.random
    
    # tf Graph Input
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    
    # Set model weights
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")
    
    # Construct a linear model
    pred = tf.add(tf.multiply(X, W), b)
    
    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    print("Starting Training")
    
    # Enable saving and restoring of model for speed
    
    # Start training
    with tf.Session() as sess:
    
        # Run the initializer
        sess.run(init)
        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})
                
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                    "W=", sess.run(W), "b=", sess.run(b))    
        
        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
    
        # Graphic display
        plt.plot(train_X, train_Y, 'ro', label='Original data')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()
    
        # Testing example, as requested (Issue #2)
        i_list = random.sample(range(1, len(train_X)), 20)
        test_X = []
        test_Y = []
        for i in i_list:
            test_X.append(train_X[i])
            test_Y.append(train_Y[i])
        test_X = numpy.asarray(test_X)
        test_Y = numpy.asarray(test_Y)
        
        
        #test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
        #test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    
        print("Testing... (Mean square loss Comparison)")
        testing_cost = sess.run(
            tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
            feed_dict={X: test_X, Y: test_Y})  # same function as cost above
        print("Testing cost=", testing_cost)
        print("Absolute mean square loss difference:", abs(
            training_cost - testing_cost))
    
        plt.plot(test_X, test_Y, 'bo', label='Testing data')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    run()
