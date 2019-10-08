# coding=utf-8

#  1.加了accu_value  和 accuracy   2.加了dropout


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ------------------------导入数据-----------------------
f = open('train_data.csv')
df = pd.read_csv(f)  # 读入负载数据
data = np.array(df['CPU'])  # 获取负载列
# data = data[::-1]  # 反转，使数据按照日期先后顺序排列

# 折线图展示data
plt.figure()
plt.plot(data)
plt.show()
normalize_data = (data - np.mean(data)) / np.std(data)  # 标准化
normalize_data = normalize_data[:, np.newaxis]  # 增加维度

# 设置训练集和参数
time_step = 20  # 时间步
rnn_unit = 10  # hidden layer units
lstm_layers = 2  # 两层神经元
batch_size = 60  # 每一批次训练多少个样例
input_size = 1  # 输入层维度
output_size = 1  # 输出层维度
lr = 0.0006  # 学习率
train_x, train_y = [], []  # 训练集
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())

# ------------定义神经网络变量-------------------
X = tf.placeholder(tf.float32, [None, time_step, input_size])  # 每批次输入网络的tensor
Y = tf.placeholder(tf.float32, [None, time_step, output_size])  # 每批次tensor对应的标签
# 处理过拟合问题。该值在其起作用的层上，给该层每一个神经元添加一个“开关”，“开关”打开的概率是keep_prob定义的值，一旦开关被关了，这个神经元的输出将被“阻断”。这样做可以平衡各个神经元起作用的重要性，杜绝某一个神经元“一家独大”，各种大佬都证明这种方法可以有效减弱过拟合的风险。
keep_prob = tf.placeholder(tf.float32, [])

# 输入层、输出层权重和偏置
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# 定义神经网络
def lstm(batch):
    w_in = weights['in']  # 获取输入层的w b
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入

    # 定义一个带着“开关”的LSTM单层，一般管它叫细胞
    def lstm_cell():
        cell = tf.nn.rnn_cell.LSTMCell(rnn_unit, reuse=tf.get_variable_scope().reuse)  # 默认：forget_bias=1.0  表示不忘记数据
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for i in range(lstm_layers)])
    init_state = cell.zero_state(batch, dtype=tf.float32)  # 初始化state
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入
    w_out = weights['out']  # 获取输出层的w b
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out  #
    return pred, final_states  # 返回预测值和最终状态


# -------------------------训练模型--------------------------
def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(batch_size)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))  # 损失函数
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)  # AdamOptimizer

    # input_y = tf.reshape(Y, [-1])
    # pre_y = tf.reshape(pred, [-1])
    # # 定义一个值用作展示训练的效果，它的定义为：选择预测值和实际值差别最大的情况并将差值返回
    # min_error = tf.reduce_mean(tf.square(pre_y - input_y))
    # # 准确率
    # # accuracy = tf.reduce_mean((tf.abs(pre_y - input_y))/(tf.maximum(pre_y, input_y)) * 100)
    # correct_prediction = tf.equal(tf.argmax(pre_y, 1), tf.argmax(input_y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练x次
        for i in range(1000):  # We can increase the number of iterations to gain better result.
            step = 0
            start = 0
            end = start + batch_size  # 60
            while (end < len(train_x)):  # 6110 - 20 = 6090
                # _, loss_, min_error_, accuracy_ = sess.run([train_op, loss, min_error, accuracy], feed_dict={X: train_x[start:end], Y: train_y[start:end], keep_prob: 1.0})
                _, loss_ = sess.run([train_op, loss],
                                                           feed_dict={X: train_x[start:end], Y: train_y[start:end],
                                                                      keep_prob: 1.0})
                start += batch_size  # 起始位置往后迭代
                end = start + batch_size
                # 每10步打印一次
                if step % 10 == 0:
                    print("Number of iterations:", i, "number of batches:", step / 10 + 1, " batch_loss:", loss_)
                    print("model_save", saver.save(sess, 'model_save1\\modle.ckpt'))
                    # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
                    # if you run it in Linux,please use  'model_save1/modle.ckpt'
                step += 1
        print("The train has finished")


train_lstm()


# ----------------------------预测模型----------------------------------
def prediction():
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(1)  # 预测时只输入[1,time_step,input_size] 的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess, 'model_save1\\modle.ckpt')
        # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        # if you run it in Linux,please use  'model_save1/modle.ckpt'
        prev_seq = train_x[-1]  # 之前的
        predict = []
        # 得到之后的100个结果
        for i in range(100):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()


prediction()
