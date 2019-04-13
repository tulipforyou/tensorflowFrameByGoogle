import tensorflow as tf
import os
from numpy.random import RandomState
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""本章主要介绍深层神经网络去线性化分析"""

def loss():
    w1 = tf.Variable(tf.random_normal([2, 1], stddev=2, seed=1))
    x = tf.placeholder(tf.float32, shape=(None, 2), name='input')
    y = tf.matmul(x, w1)
    y_ = tf.placeholder(tf.float32, shape=(None, 1))  # 1

    loss_less=10
    loss_more=1
    loss=tf.reduce_sum(tf.where(tf.greater(y,y_),
                                                (y-y_)*loss_more,
                                                (y_-y)*loss_less))

    global_steps = tf.Variable(0)
    learaning_rate = tf.train.exponential_decay(0.1, global_steps, 100, 0.96, staircase=False)
    #指数衰减性自动设置学习率，初始值为0.1,每100轮后乘以0.96,global_steps在训练时自动更新
    train_step = tf.train.AdamOptimizer(learaning_rate).minimize(loss,global_step=global_steps)

    rdm = RandomState(1)
    X = rdm.rand(128, 2)
    Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("w1 shape: ", w1.get_shape(), '\n', sess.run(w1))
        for i in range(20000):
            start = (i * 8) % 128
            end = min(start + 8, 128)
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
            if i % 1000 == 0:
                result = sess.run(loss, feed_dict={x: X, y_: Y})
                print("第%d次训练之后交叉熵为： %g " % (i + 1000, result))
                print("此时学习率为： ",sess.run(learaning_rate),'\n')

        print("训练后的W1,W2为：\n")
        print(sess.run(w1), '\n')


if __name__=='__main__':
    loss()
