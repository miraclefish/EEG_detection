import tensorflow as tf
import numpy as np


# 定义LSTM模型
class LSTM_Model:
    def __init__(self, input_shape, state_size, batch_size, learning_rate, batch_norm=False):

        self.input_shape = input_shape
        self.state_size = state_size
        self.num_layers = 1
        self.time_steps = self.input_shape[0]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold = 0.7
        self.batchnorm = batch_norm
        self.initial = tf.random_normal_initializer(mean=0, stddev=1)
        self.weight, self.bias = self._get_weight_bias()
        self.inputs, self.labels, self.global_step = self._inputs()
        self.conv_out = self._CNNlayer()
        self.output, self.state = self._LSTMlayer()
        self.state_out = self._state_out()
        self.loss = self._loss_define()
        self.prediction = self._prediction()
        self.accuracy = self._accuracy_define()
        self.recall = self._recall_define()
        self.precision = self._precision_define()
        # self.learning_rate = tf.train.exponential_decay(learning_rate, global_step=self.global_step, decay_rate=0.9, decay_steps=10)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def _get_weight_bias(self):
        weight = tf.get_variable("weight", [self.state_size, 1], dtype=tf.float32, initializer=self.initial)
        bias = tf.get_variable("bias", [1], dtype=tf.float32, initializer=self.initial)
        return weight, bias

    def _inputs(self):
        inputs = tf.placeholder(
            dtype=tf.float32,
            shape=(None, self.input_shape[0], self.input_shape[1]),
            name="inputs",
        )
        # labels = tf.placeholder(dtype=tf.float32, shape=(None, self.input_shape[0]), name="labels")
        labels = tf.placeholder(dtype=tf.float32, shape=(None, 27), name="labels")
        global_step = tf.placeholder(dtype=tf.int32, name="global_step")
        return inputs, labels, global_step

    def _CNNlayer(self):
        epsilon = 1e-3
        f1 = tf.get_variable("filter1", [5, self.input_shape[1], 32], dtype=tf.float32, initializer=self.initial)
        conv1 = tf.nn.conv1d(self.inputs, filters=f1, stride=3, name="conv_1", padding="VALID")
        if self.batchnorm == True:
            mean1, var1 = tf.nn.moments(conv1, axes = [0,1])
            conv1 = tf.nn.batch_normalization(conv1, mean=mean1, variance=var1, offset=None, scale=None, variance_epsilon=epsilon)
        conv1_pool = tf.nn.max_pool1d(conv1, ksize=[2,1,1], strides=[2,1,1], padding="VALID")
        conv1 = tf.nn.tanh(conv1_pool)
        f2 = tf.get_variable("filter2", [5, 32, 64], dtype=tf.float32, initializer=self.initial)
        conv2 = tf.nn.conv1d(conv1, filters=f2, stride=3, name="conv_2", padding="VALID")
        if self.batchnorm == True:
            mean2, var2 = tf.nn.moments(conv2, axes = [0,1])
            conv2 = tf.nn.batch_normalization(conv2, mean=mean2, variance=var2, offset=None, scale=None, variance_epsilon=epsilon)
        conv2_pool = tf.nn.max_pool1d(conv2, ksize=[3,1,1], strides=[2,1,1], padding="VALID")
        conv2 = tf.nn.tanh(conv2_pool)
        f3 = tf.get_variable("filter3", [8, 64, 128], dtype=tf.float32, initializer=self.initial)
        conv3 = tf.nn.conv1d(conv2, filters=f3, stride=5, name="conv_3", padding="VALID")
        if self.batchnorm == True:
            mean3, var3 = tf.nn.moments(conv3, axes = [0,1])
            conv3 = tf.nn.batch_normalization(conv3, mean=mean3, variance=var3, offset=None, scale=None, variance_epsilon=epsilon)
        conv3 = tf.nn.tanh(conv3)
        return conv3

    def _LSTMlayer(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_size)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
        output, state = tf.nn.dynamic_rnn(cell, inputs=self.conv_out, dtype=tf.float32)
        return output, state

    def _state_out(self):
        state_out = (
            tf.matmul(tf.reshape(self.output, [-1, self.state_size]), self.weight)
            + self.bias
        )
        # state_out = tf.nn.sigmoid(state_out)
        state_out = tf.reshape(state_out, [-1, self.conv_out.get_shape().as_list()[-2]])
        # state_out = tf.reshape(state_out, [-1, self.input_shape[0]])
        return state_out

    def _loss_define(self):
        loss_op = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=self.state_out, labels=self.labels, pos_weight=20))
        return loss_op

    def _accuracy_define(self):
        out = self.state_out
        one = tf.ones_like(out)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.labels), "float"))
        return accuracy

    def _recall_define(self):
        out = self.state_out
        one = tf.ones_like(out)
        tp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(tf.equal(self.prediction, one), tf.equal(self.labels, one)),
                "float",
            )
        )
        P_label = tf.reduce_sum(tf.cast(tf.equal(self.labels, one), "float"))
        recall = tp / P_label
        return recall

    def _precision_define(self):
        out = self.state_out
        one = tf.ones_like(out)
        tp = tf.reduce_sum(
            tf.cast(
                tf.logical_and(tf.equal(self.prediction, one), tf.equal(self.labels, one)),
                "float",
            )
        )
        P_pred = tf.reduce_sum(tf.cast(tf.equal(self.prediction, one), "float"))
        precision = tp / P_pred
        return precision

    def _prediction(self):
        out = self.state_out
        one = tf.ones_like(out)
        zero = tf.zeros_like(out)
        prediction = tf.where(out < self.threshold, x=zero, y=one)
        return prediction
