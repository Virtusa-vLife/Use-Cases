import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import tensorflow as tf
seed=128

class ANN(object):
    
    def __init__(self):
        learning_rate=0.001
        training_epochs=20000
        display_step=50
        n_samples=6400
        
        tf.reset_default_graph()
        self.x=tf.placeholder(tf.float32,[None,124],name='1')
        
        # hidden layer 1
        self.weights_hidden1 = tf.Variable(tf.random_normal([124, 64]),name='2')
        self.bias_hidden1 = tf.Variable(tf.random_normal([64]),name='3')
        self.preactivations_hidden1 = tf.add(tf.matmul(self.x, self.weights_hidden1), self.bias_hidden1)
        self.activations_hidden1 = tf.nn.softmax(self.preactivations_hidden1)

        # hidden layer 2                          
        self.weights_hidden2 = tf.Variable(tf.random_normal([64, 128]),name='4')
        self.bias_hidden2 = tf.Variable(tf.random_normal([128]),name='5')
        self.preactivations_hidden2 = tf.add(tf.matmul(self.activations_hidden1, self.weights_hidden2), self.bias_hidden2)
        self.activations_hidden2 = tf.nn.softmax(self.preactivations_hidden2)


        # hidden layer 3                          
        self.weights_hidden3 = tf.Variable(tf.random_normal([128, 64]),name='6')
        self.bias_hidden3 = tf.Variable(tf.random_normal([64]),name='7')
        self.preactivations_hidden3 = tf.add(tf.matmul(self.activations_hidden2, self.weights_hidden3), self.bias_hidden3)
        self.activations_hidden3 = tf.nn.softmax(self.preactivations_hidden3)

        # output layer
        self.weights_output = tf.Variable(tf.random_normal([64, 2]),name='8')
        self.bias_output = tf.Variable(tf.random_normal([2]),name='9') 
        self.preactivations_output = tf.add(tf.matmul(self.activations_hidden3, self.weights_output), self.bias_output)
        self.y=tf.nn.softmax(self.preactivations_output)
        
        self.y_=tf.placeholder(tf.float32,[None,2],name='10')
        self.cost=tf.reduce_sum(tf.pow(self.y_-self.y,2))/(2*n_samples)
        self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
        
        
    def load_model(self,sess,path):
        saver = tf.train.Saver()

        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        self.sess=sess
            # Restore variables from disk.
        saver.restore(self.sess, path)
        print("Model restored.")
        
    def predict(self,test):
        self.sess=sess
        self.sess.run(self.y,feed_dict={x:test})
            
    def accuracy(self,pred,pred1):
        res=[]
        for i in pred:
            i = np.where(i > 0.5, 1, 0)
            res.append(i[0])
            res1=[]
        
        for i in inputY_test:
            i = np.where(i > 0.5, 1, 0)
            res1.append(i[0])
        return accuracy_score(res,res1)
        
    def score(self,inputX,inputY):
        self.sess=sess
        return self.sess.run(self.cost,feed_dict={x:inputX,y_:inputY})