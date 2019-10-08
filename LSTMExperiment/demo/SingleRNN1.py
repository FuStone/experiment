import tensorflow as tf
import numpy as np

# =====================一、学习单步的RNN：RNNCell ======================
# 1.每调用一次RNNCell的call方法，就相当于在时间上“推进了一步”，这就是RNNCell的基本功能。
# 2.RNNCell只是一个抽象类，我们用的时候都是用的它的两个子类BasicRNNCell和BasicLSTMCell。顾名思义，前者是RNN的基础类，后者是LSTM的基础类。
# 3.除了call方法外，对于RNNCell，还有两个类属性比较重要：
    # state_size
    # output_size
    # 前者是隐层的大小，后者是输出的大小。
# 4.比如我们通常是将一个batch送入模型计算，设输入数据的形状为(batch_size, input_size)，那么计算时得到的隐层状态就是(batch_size, state_size)，输出就是(batch_size, output_size)。

# =============================== 学习如何一次执行多步：tf.nn.dynamic_rnn ==========================
# TensorFlow提供了一个tf.nn.dynamic_rnn函数，使用该函数就相当于调用了n次call函数。即通过{h0,x1, x2, …., xn}直接得{h1,h2…,hn}。

# --------------------------BasicRNNCell-----------------------------
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)    # state_size = 128
print(cell.state_size)   # 128

# inputs = tf.placeholder(np.float32, shape=(32, 100))  # 32是batch_size
# h0 = cell.zero_state(32, np.float32)  # 通过zero_state 得到一个全为0的初始状态，形状为（batch_size,state_size）
# # output, h1 = cell.call(inputs, h0)  # 调用call函数   报错：AttributeError: 'BasicRNNCell' object has no attribute '_kernel'
# output, h1 = cell.__call__(inputs, h0)  # 调用call函数
#
# print(h1.shape)   # (32, 128)


# ----------------------------BasicLSTMCELL-------------------------
# LSTM可以看做有两个隐藏状态：h和c，对应的隐层就是一个Tuple，每个都是（batch_size,state_size）的形状

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32, 100))  # 32是batch_size
h0 = lstm_cell.zero_state(32, np.float32)   # 通过zero_state得到一个全0的初始状态
output, h1 = lstm_cell.__call__(inputs,h0)

# =================dynamic_rnn===============
# 输入数据的格式为(batch_size, time_steps, input_size)，其中time_steps表示序列本身的长度，如在Char RNN中，长度为10的句子对应的time_steps就等于10。
# 最后的input_size就表示输入数据单个序列单个时间维度上固有的长度
inputs_new = tf.placeholder(np.float32, shape=(32, 10, 100))   # 每一个batch有32，共有10step，每一个输入长100
init_state = lstm_cell.zero_state(32, np.float32)
outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs_new, initial_state=init_state, dtype=tf.float32)

print(h1.h)  # shape = (32,128)
print(h1.c)  # shape = (32,128)

print(outputs)
# print(state)



