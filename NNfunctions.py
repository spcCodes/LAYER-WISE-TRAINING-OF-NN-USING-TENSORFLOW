#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 01:05:13 2017

@author: suman
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy.linalg import inv

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32 , shape=[n_x , None] , name ="X")
    Y = tf.placeholder(tf.float32 , shape=[n_y , None] , name = "Y")
    ### END CODE HERE ###
    
    return X, Y

# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x , n_h , n_y ):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [n_h, n_x]
                        b1 : [n_h, 1]
                        W2 : [n_y, n_h]
                        b2 : [n_y, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [n_h, n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [n_h, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [n_y, n_h], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [n_y, 1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X), b1)                                           # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1), b2)                                             # Z2 = np.dot(W2, a1) + b2
    ### END CODE HERE ###
    
    return Z2 , A1 


# GRADED FUNCTION: compute_cost 

def compute_cost(Z2, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z2)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits , labels= labels))
    ### END CODE HERE ###
    
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 50, beta = 0.01 ,print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    n_h = int((2*(n_x+n_y))/3)
    #n_h = n_x + 50
    costs = []                                       # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x , n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters(n_x , n_h , n_y)
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z2 , A1 = forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z2, Y)
    ### END CODE HERE ###
    
    ###ADDING L2 REGULARIZATION###
    # Loss function with L2 Regularization with beta=0.01
    #regularizers = tf.nn.l2_loss(parameters['W1']) + tf.nn.l2_loss(parameters['W2'])
    #cost = tf.reduce_mean(cost + beta * regularizers)
    
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
# =============================================================================
#         # plot the cost
#         plt.plot(np.squeeze(costs))
#         plt.ylabel('cost')
#         plt.xlabel('iterations (per tens)')
#         plt.title("Learning rate =" + str(learning_rate))
#         plt.show()
# =============================================================================

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        TrAccu=  accuracy.eval({X: X_train, Y: Y_train})
        TeAccu=  accuracy.eval({X: X_test, Y: Y_test})
        TrLoss = sess.run( cost, feed_dict={X: X_train, Y: Y_train})
        TeLoss = sess.run( cost, feed_dict={X: X_test, Y: Y_test})
        return parameters , TrAccu , TeAccu , TrLoss , TeLoss
    
def DataSeperation(Data,Labels , PercentageSeperation):
    unique, counts = np.unique(Labels, return_counts=True)
    TrainDataSet=[]
    TestDataSet=[]
    for i in range(len(unique)):
        searchValue=unique[i]
        indices = np.where(Labels==searchValue)
        LengthIndices=len(indices[0])
        Cluster=[]
        for j in range(0,LengthIndices):
            Cluster.append(Data[indices[0][j]])
        TrainCut= math.ceil((PercentageSeperation/100)*LengthIndices)
        Cluster=np.matrix(Cluster)
        Cluster=np.random.permutation(Cluster)
        TrainSet = Cluster[0:TrainCut,:]
        TestSet = Cluster[TrainCut:LengthIndices , :]
        
        Label1= np.ones((TrainCut,1))*i
        Label2 = np.ones((LengthIndices-TrainCut,1))*i
        Train= np.concatenate((TrainSet,Label1),axis=1)
        Test = np.concatenate((TestSet,Label2),axis=1)
        
        TrainDataSet.append(Train)
        TestDataSet.append(Test)
        
        Train=[]
        Test=[]
        Label1=[]
        Label2=[]
        Cluster=[]
        indices=[]
        
    return TrainDataSet , TestDataSet

def myencoder(Labels):
    '''perform the one hot encoding for y data'''
    Labels=Labels.reshape(Labels.shape[0],-1)
    onehotencoder=OneHotEncoder(categorical_features=[0])
    LabelsEncoded = onehotencoder.fit_transform(Labels).toarray()
    return LabelsEncoded

def GenerateDataset(Xinput,Yinput,Percentage):
    training,testing=DataSeperation(Xinput,Yinput,Percentage)
    training=np.concatenate(training , axis=0) ## converting list of arrays to array --> training data
    trainingLabels=training[:,-1]
    trainingData=training[:,:-1]
    testing=np.concatenate(testing , axis=0) ## converting list of arrays to array --> testing data
    testingLabels=testing[:,-1]
    testingData=testing[:,:-1]
    
    trainingLabelsEncoded= myencoder(trainingLabels)
    testingLabelsEncoded = myencoder(testingLabels)
    X_train=trainingData.T
    X_test=testingData.T
    Y_train=trainingLabelsEncoded.T
    Y_test=testingLabelsEncoded.T
    
    return X_train,Y_train,X_test,Y_test

def mysolve(A):
    u, s, v = np.linalg.svd(A)
    Ainv = np.dot (v.transpose(), np.dot(np.diag(s**-1),u.transpose()))
    return(Ainv)

def forward_propagationmod(X, W1 , b1):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X), b1)                                           # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    #Z2 = tf.add(tf.matmul(W2,A1), b2)                                             # Z2 = np.dot(W2, a1) + b2
    ### END CODE HERE ###
    
    return A1

def initialize_parameters1(n_x , n_h , n_y ):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [n_h, n_x]
                        b1 : [n_h, 1]
                        W2 : [n_y, n_h]
                        b2 : [n_y, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [n_h, n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [n_h, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [n_y, n_h], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [n_y, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W2", [n_y, n_h], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b2", [n_y, 1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters