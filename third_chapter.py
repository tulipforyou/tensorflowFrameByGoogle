import tensorflow as tf
import os
from numpy.random import RandomState
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def many_graph():
    g1 = tf.Graph()
    with g1.as_default():
        v = tf.Variable(tf.random_normal([2, 6], stddev=2))  # 正太分布，标准差为2
        print(v)

    g2 = tf.Graph()
    with g2.as_default():
        v = tf.Variable(tf.constant(0))
        print(v)


def tensor_test():
    """a = tf.constant([[[2, 6.0], [9, 6]]], tf.float32, name='a')
    b = tf.constant([[[9, 6], [9, 5]]], tf.float32, name='b')
    result = tf.multiply(a, b, name='add')  # name为node名称

    print(result)

    with tf.Session() as sess:
        print(sess.run(result))"""

    w1 = tf.Variable(tf.random_normal([2, 6], stddev=2, seed=1)) #2 3
    w2 = tf.Variable(tf.random_normal([6, 1], stddev=2, seed=1)) #3,1
    x = tf.placeholder(tf.float32, shape=(None, 2), name='input')
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)
    y_=tf.placeholder(tf.float32,shape=[None,1]) # 1
    """其中y_代表正确结果，y代表预测结果。
    tf.clip_by_value函数可以将一个张量中的数值限制在一个范围内,
    小于下限的值替换为下限，大于上限的值替换为上限"""
    #tensorflow中交叉熵的实现
    cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

    rdm = RandomState(1)
    X = rdm.rand(1280, 2)
    Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("w1 shape: ", w1.get_shape(), '\n', sess.run(w1))
        print("w2 shape: ", w2.get_shape(), '\n', sess.run(w2))
        for i in range(100000):
            start=(i*8)%1280
            end=min(start+8,1280)
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
            if i%1000==0:
                result=sess.run(cross_entropy, feed_dict={x: X, y_: Y})
                print("第%d次训练之后交叉熵为： %g "%(i+1000,result))

        print("训练后的W1,W2为：\n")
        print(sess.run(w1),'\n')
        print(sess.run(w2))



if __name__ == '__main__':
    many_graph()
    tensor_test()
