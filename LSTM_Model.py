import tensorflow as tf
import numpy as np


# 定义LSTM模型
class LSTM_Model:
    def __init__(self, input_shape, state_size, batch_size, learning_rate):

        self.input_shape = input_shape
        self.state_size = state_size
        self.num_layers = 1
        self.threshold = 0.7
        self.time_steps = self.input_shape[0]
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.initial = tf.random_normal_initializer(mean=0, stddev=1)
        self.weight, self.bias = self._get_weight_bias()
        self.inputs, self.labels = self._inputs()
        self.output, self.state = self._LSTMlayer()
        self.state_out = self._state_out()
        self.loss = self._loss_define()
        self.prediction = self._prediction()
        self.accuracy = self._accuracy_define()
        self.recall = self._recall_define()
        self.precision = self._precision_define()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
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
        labels = tf.placeholder(
            dtype=tf.float32, shape=(None, 19), name="labels"
        )
        return inputs, labels

    def _LSTMlayer(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.state_size)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
        output, state = tf.nn.dynamic_rnn(cell, inputs=self.inputs, dtype=tf.float32)
        return output, state

    def _state_out(self):
        state_out = (
            tf.matmul(tf.reshape(self.output, [-1, self.state_size]), self.weight)
            + self.bias
        )
        # state_out = tf.nn.sigmoid(state_out)
        state_out = tf.reshape(state_out, [-1, self.input_shape[0]])
        return state_out

    def _loss_define(self):
        loss_op = tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                logits=self.state_out, labels=self.labels, pos_weight=5
            )
        )
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