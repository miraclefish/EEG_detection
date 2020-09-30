import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from LSTM_Model import Model
from DataLoader import Dataloader


class Trainer:
    def __init__(self, batch_size):
        self.Model = Model(
            input_shape=(5000, 23),
            state_size=128,
            batch_size=batch_size,
            learning_rate=0.001,
        )
        self.batch_size = batch_size

    def train(self, num_epochs=10):

        data = Dataloader(batchsize=self.batch_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            
            train_X, _ = data.getTrainData()
            num_tr_iter = int(train_X.shape[0] / self.batch_size)
            global_step = 0
            Loss = []
            Acc = []
            for epoch in range(num_epochs):
                print("-------------------------------")
                print("Training epoch: {}".format(epoch + 1))
                data.i = 0
                acc = np.zeros(num_tr_iter)
                loss = np.zeros(num_tr_iter)
                for iteration in range(num_tr_iter):
                    global_step += 1
                    x_batch, y_batch = data.nextBatch()
                    # Run optimization op (backprop)
                    feed_dict_batch = {
                        self.Model.inputs: x_batch,
                        self.Model.labels: y_batch,
                    }
                    sess.run(self.Model.train_op, feed_dict=feed_dict_batch)

                    if iteration % 3 == 0:
                        # Calculate and display the batch loss and accuracy
                        loss_batch, acc_batch = sess.run(
                            [self.Model.loss, self.Model.accuracy],
                            feed_dict=feed_dict_batch,
                        )
                        print(
                            "iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".format(
                                iteration, loss_batch, acc_batch
                            )
                        )
                        acc[iteration] = acc_batch
                        loss[iteration] = loss_batch
                x_valid, y_valid = data.getValiData()
                val_loss, val_acc = sess.run(
                    [self.Model.loss, self.Model.accuracy],
                    feed_dict={self.Model.inputs: x_valid, self.Model.labels: y_valid},
                )
                print(
                    "Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".format(
                        epoch + 1, val_loss, val_acc
                    )
                )
                acc, loss = np.mean(acc), np.mean(loss)
                Loss.append(loss)
                Acc.append(acc)

            x_test, y_test = data.getTestData()
            feed_dict_test = {self.Model.inputs: x_test, self.Model.labels: y_test}
            test_loss, test_acc = sess.run(
                [self.Model.loss, self.Model.accuracy], feed_dict=feed_dict_test
            )
            print("--------------------")
            print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(test_loss, test_acc))
        return Loss, Acc


train = Trainer(batch_size=32)
Loss, Acc = train.train()

pass

