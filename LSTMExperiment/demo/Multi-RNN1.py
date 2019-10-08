# ======================== 三、学习如何堆叠RNNCell：MultiRNNCell ========================
# 1.单层RNN的能力有限，我们需要多层的RNN。将x输入第一层RNN的后得到隐层状态h，这个隐层状态就相当于第二层RNN的输入，
    # 第二层RNN的隐层状态又相当于第三层RNN的输入，以此类推。
# 2.在TensorFlow中，可以使用tf.nn.rnn_cell.MultiRNNCell函数对RNNCell进行堆叠
# 3.通过MultiRNNCell得到的cell并不是什么新鲜事物，它实际也是RNNCell的子类，因此也有call方法、state_size和output_size属性。
# 4.同样可以通过tf.nn.dynamic_rnn来一次运行多步。
import tensorflow as tf
import numpy as np

# 每调用一次这个函数就返回一个BasicRNNCell
def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# 用tf.nn.rnn_cell MultiRnnCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])  # 3层RNN
# 得到的cell实际也是RNNCell的子类
# 他的state_size是(128,128,128)    注意：并不是128x128x128的意思，而是表示有3个隐层状态，每个隐层状态的大小为128
print(cell.state_size)   # (128,128,128)
inputs = tf.placeholder(np.float32, shape=(32, 100))   # 32 是 batch_size
# 使用对应的call函数
h0 = cell.zero_state(32, np.float32)  # 通过zero_state得到一个全0的初始状态
output, h1 = cell.__call__(inputs, h0)
print(h1)  # tuple中含有3个32x128的向量 分别是cell1, cell2, cell3的
# print(output)

# 同样可以通过tf.nn.dynamic_rnn来一次运行多步。
inputs_new = tf.placeholder(np.float32, shape=(32, 10, 100))   # 每一个batch有32，共有10step，每一个输入长100
init_state = cell.zero_state(32, np.float32)
outputs, state = tf.nn.dynamic_rnn(cell, inputs_new, initial_state=init_state, dtype=tf.float32)
print(state)  # tuple中含有3个32x128的向量
print(outputs)   # Tensor (32, 10, 128)






