import time
import abc

import numpy as np
import tensorflow as tf


class NN:
    def __init__(self, x_train, y_train, x_val, y_val, print_mode=True):
        #  Reset previous graphs
        tf.reset_default_graph()

        #  Model Data 
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        # Print mode (text | plot)
        self.print_mode = print_mode

    @abc.abstractmethod
    def create_graph(self):
        return

    def print_trainable_parameters(self):
        params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Trainable parameters {}'.format(params))

    def train(self, epochs=100, batch_size=256, patience=25):
        training_start_time = time.time()

        train_accuracies = []
        test_accuracies = []

        worse_epoch_count = 0
        best_train_accuracy = 0
        best_test_accuracy = 0

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            for epoch in range(1, epochs + 1):
                if self.print_mode:
                    progress = tf.keras.utils.Progbar(target=len(self.x_train),
                                                      stateful_metrics=['batch loss', 'time'],
                                                      width=30, interval=0.5)

                start_time = time.time()
                if self.print_mode:
                    print('> Epoch {}:'.format(epoch))

                for batch_idx, batch in enumerate(
                        self._next_batch(self.x_train, self.y_train, batch_size, shuffle=True)):
                    features, targets = batch
                    d = {self.names: features, self.genders: targets}
                    loss, _ = sess.run([self.loss, self.optimize], feed_dict=d)

                    if self.print_mode:
                        elapsed_time = time.time() - start_time
                        progress.update(batch_idx * batch_size,
                                        values=[('time', elapsed_time), ('batch loss', loss), ('epoch loss', loss)])

                if self.print_mode:
                    progress.update(len(self.x_train), values=[('batch loss', loss), ('epoch loss', loss)])

                train_accuracy = sess.run(self.accuracy,
                                          feed_dict={self.names: self.x_train, self.genders: self.y_train})
                test_accuracy = sess.run(self.accuracy, feed_dict={self.names: self.x_val, self.genders: self.y_val})

                if self.print_mode:
                    print('Epoch {:2} | Training set accuracy = {:.4f}, Test set accuracy = {:.4f}'
                          .format(epoch, train_accuracy, test_accuracy))

                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)

                best_train_accuracy = max(train_accuracy, best_train_accuracy)
                if test_accuracy > best_test_accuracy:
                    worse_epoch_count = 0
                    best_test_accuracy = test_accuracy
                else:
                    worse_epoch_count += 1
                    if worse_epoch_count == patience:
                        if self.print_mode:
                            print('Early stopping at epoch {}'.format(epoch))
                        break

        return best_train_accuracy, train_accuracies, best_test_accuracy, test_accuracies, time.time() - training_start_time

    def _next_batch(self, x, y, batch_size, shuffle=False):
        position = 0
        while position + batch_size < len(x):
            offset = position + batch_size
            yield x[position:offset], y[position:offset]
            position = offset
        yield x[position:], y[position:]

    def _create_graph_input(self, input_dim, vocab_dim, emb_dim):
        # Placeholders for input and targets
        self.names = tf.placeholder(tf.int32, shape=[None, input_dim], name='Names')
        self.genders = tf.placeholder(tf.float32, shape=[None, 1], name='Genders')

        #  Embedding Matrix (0-pad is not a variable, remains 0)
        padding_vector = tf.zeros(shape=(1, emb_dim), dtype=tf.float32, name='ZeroPadding')
        symbol_embedding = tf.get_variable('W', shape=(vocab_dim, emb_dim), dtype=tf.float32)
        symbol_embedding = tf.concat([padding_vector, symbol_embedding], axis=0)

        # Word embeddings
        return tf.nn.embedding_lookup(symbol_embedding, self.names)

    def _create_graph_output(self, last_layer, lr):
        #  Dense layer with binary output
        logits = tf.layers.dense(last_layer, 1)

        #  Loss & Optimization
        logits_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.genders)
        self.loss = tf.reduce_mean(logits_loss)
        optimizer = tf.contrib.opt.LazyAdamOptimizer(lr)
        self.optimize = optimizer.minimize(self.loss)

        #  Prediction & Accuracy
        self.predictions = tf.round(tf.sigmoid(logits))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.genders), dtype=tf.float32))
