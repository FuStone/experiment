import numpy as np
import tensorflow as tf


l1 = [21, 34, 45]
l2 = [55, 25, 77]
l = np.average(np.abs(l1 - l2))
print(l)
# v = list()
# l_min = tf.reduce_min(l)
# min = tf.reshape(l_min, [-1, 1])
# min1 = tf.reshape(min, [-1])
#
# min_ = tf.reshape(tf.reduce_min(l), [-1])
#
#
# res = np.min(l)
#
# sess = tf.Session()
# # print(sess.run(min))
# # print(sess.run(min1))
# print(min_)
#
# print(sess.run())
# print(sess.run(res))

