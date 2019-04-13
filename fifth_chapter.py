import tensorflow as tf
import os
from numpy.random import RandomState
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""MNIST数字识别问题"""

def get_mnist_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
    return mnist

def show_mnist_data():
    mnist=get_mnist_data()
    print("Training data size: ",mnist.train.num_examples)
    print("Validating data size: ", mnist.validation.num_examples)
    print("Testing data size: ", mnist.test.num_examples)
    print("Example training data: ", mnist.train.images[0])
    print("Example training data label: ", mnist.train.labels[4])

def save_graph():
    v1=tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
    v2=tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
    result=tf.add(v1,v2)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.save(sess,'model.ckpt')



if __name__=='__main__':
    get_mnist_data()
    show_mnist_data()
    save_graph()
