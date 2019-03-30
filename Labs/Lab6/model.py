import logging
import string
from typing import Dict, Generator, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

PAD_EXPR = '<PAD>'


class WordTagEstimator:
    """
    Word Tag Estimation using CNN as described in https://arxiv.org/pdf/1511.08308.pdf
    """

    def __init__(self, train_data_path: str, test_data_path: str):
        self.train_X, self.train_Y, self.input_dim = WordTagEstimator._extract_data(train_data_path)
        self.test_X, self.test_Y, _ = WordTagEstimator._extract_data(test_data_path)

        self.train_X, self.train_Y = self._pad_sequences_words(pad=PAD_EXPR)
        self.test_X, self.test_Y = self._pad_sequences_words(pad=PAD_EXPR, test=True)
        self.pos_tags = set(self.train_Y)

        # All possible vocabulary letters, numbers and punctuation symbols
        self.vocabulary = list(string.ascii_letters + string.digits + string.punctuation)

        # Mapping character to index
        # Add padding as special character outside the vocabulary (ID=0)
        index = 1
        self.char2idx: Dict[str, int] = {PAD_EXPR: 0}
        for c in self.vocabulary:
            self.char2idx[c] = index
            index += 1

        # Mapping POS to index
        index = 0
        self.pos2idx: Dict[str, int] = {}
        for pos in self.pos_tags:
            self.pos2idx[pos] = index
            index += 1

    def create_graph(self, emb_dim: int, hid_dim: int, conv_win_size: int, summary_path='./tf_summary'):
        """
        Create TensorFlow Computational Graph.
        Also stores a Summary of the generated graph in <summary_path> for Tensorboard visualization.
        :param emb_dim: Dimension of the embedded vectors.
        :param hid_dim: Number of convolution kernels.
        :param conv_win_size: Window size for the convolution operation.
        :param summary_path: Path to write the graph summary
        """
        logging.info('Building CG...')

        self.X = tf.placeholder(tf.int64, shape=[None, self.input_dim], name='Words')
        self.Y = tf.placeholder(tf.int64, shape=None, name='POS')
        self.LR = tf.placeholder(tf.float32, name='LR')

        # Weight Matrices
        padding_vector = tf.zeros(shape=(1, emb_dim), dtype=tf.float32, name='ZeroPadding')
        symbol_embedding = tf.Variable(tf.random_uniform(shape=[len(self.vocabulary), emb_dim],
                                                         minval=-0.5, maxval=0.5, name='W'))
        symbol_embedding = tf.concat([padding_vector, symbol_embedding], axis=0)

        # Lookups
        embeddings = tf.nn.embedding_lookup(symbol_embedding, self.X)

        # Convolution operation
        conv = tf.layers.conv2d(tf.expand_dims(embeddings, axis=2),
                                filters=hid_dim, kernel_size=(emb_dim, conv_win_size),
                                activation='relu', padding='same')

        # Pooling over sequence dimension
        conv_max = tf.reshape(conv, shape=(-1, self.input_dim, hid_dim))
        conv_max = tf.reduce_max(conv_max, axis=1)

        #  Dense Layer 2 with #POS neurons (our final output)
        output_layer = tf.layers.dense(conv_max, units=len(self.pos2idx), activation=None)

        # Cross-Entropy as loss function
        y_oh = tf.one_hot(self.Y, len(self.pos2idx))
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_oh, logits=output_layer)
        self.loss = tf.reduce_mean(cross_entropy)

        # Optimizer
        optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.LR)
        self.objective = optimizer.minimize(self.loss)

        # Accuracy
        self.predictions = tf.argmax(output_layer, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, self.Y), dtype=tf.float32))

        # Store CG
        WordTagEstimator._store_tensorboard_visualization(summary_path)

        logging.info('CG built!')

    def train(self, epochs: int, minibatch_size: int, learning_rate: float, patience=2):
        """
        Trains the model displaying the progress and showing the confusion matrix for each epoch.
        :param epochs: Number of epochs to train the model.
        :param minibatch_size: Size of the minibatch for the optimizer step.
        :param learning_rate: Learning rate for the optimizer.
        :param patience: Epochs to wait until early stopping.
        """
        logging.info('Starting to train...')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            lr_annealing = self.lr_scheduler(learning_rate)
            best_test_accuracy = 0
            pat = 0

            for epoch in range(1, epochs + 1):
                print(f'\nEpoch {epoch}:')
                progress = tf.keras.utils.Progbar(target=len(self.train_X),
                                                  stateful_metrics=['batch loss'], width=50, interval=0.5)

                acc, loss = 0, 0
                lr = next(lr_annealing)
                for batch_idx, (features, labels) in enumerate(self._create_minibatches(minibatch_size)):
                    d = {self.X: features, self.Y: labels, self.LR: lr}
                    loss, _, acc = sess.run([self.loss, self.objective, self.accuracy], feed_dict=d)

                    progress.update(batch_idx * minibatch_size,
                                    values=[('accuracy', acc), ('batch loss', loss), ('epoch loss', loss)])

                progress.update(len(self.train_X),
                                values=[('accuracy', acc), ('batch loss', loss), ('epoch loss', loss)])

                features = list(map(lambda w: list(map(lambda c: self.char2idx[c], w)), self.test_X))
                labels = list(map(lambda p: self.pos2idx[p], self.test_Y))
                acc = sess.run(self.accuracy, feed_dict={self.X: features, self.Y: labels})

                print(f'Test accuracy: {acc}')
                if acc > best_test_accuracy:
                    best_test_accuracy = acc
                    pat = 0
                else:
                    pat += 1

                if pat == patience:
                    print(f'Best accuracy obtained: {best_test_accuracy}')
                    break

            self._display_confusion_matrix(sess)

    def _create_minibatches(self, minibatch_size: int) -> Generator[Tuple[List[List[int]], List[int]], None, None]:
        """
        Generates minibatches for training the model.
        :param minibatch_size: Size of the minibatch for the optimizer step.
        :return: Yields padded words of size [BATCH_SIZE, max_len_word] and POS of size [BATCH_SIZE]
        """
        batch_x, batch_y = [], []
        for idx, word in enumerate(self.train_X):
            batch_x.append(list((map(lambda x: self.char2idx[x], word))))
            batch_y.append(self.pos2idx[self.train_Y[idx]])
            if (idx + 1) % minibatch_size == 0:
                yield batch_x, np.array(batch_y)
                batch_x, batch_y = [], []
        yield batch_x, np.array(batch_y)

    def _pad_sequences_words(self, pad: str, test=False) -> Tuple[List[str], List[str]]:
        """
        Splits sentences into words and adds padding to make them all of length max_len_word.
        :param pad: Padding character (shouldn't be one of the vocabulary symbols.
        :return: List of words padded and another list of their corresponding POS tag.
        """
        sentences = self.train_X if not test else self.test_X
        pos = self.train_Y if not test else self.test_Y

        padded_words = []
        padded_words_pos = []
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                word = word[:self.input_dim]
                padded_word = [pad] * self.input_dim
                splitted_word = list(word)
                left_pad = (self.input_dim - len(splitted_word)) // 2
                padded_word[left_pad:left_pad + len(splitted_word)] = splitted_word

                padded_words.append(padded_word)
                padded_words_pos.append(pos[i][j])

        return padded_words, padded_words_pos

    def _display_confusion_matrix(self, sess: tf.Session):
        """
        Displays confusion matrix of a given epoch for the POS tagging.
        :param sess: TensorFlow session
        """
        true_pos = list(map(lambda p: self.pos2idx[p], self.test_Y))
        d = {self.X: list(map(lambda w: list(map(lambda c: self.char2idx[c], w)), self.test_X)), self.Y: true_pos}
        pred_pos = sess.run(self.predictions, feed_dict=d)

        cm = confusion_matrix(true_pos, pred_pos)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
        ax.grid(False)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=self.pos_tags, yticklabels=self.pos_tags,
               title='Test confusion Matrix', ylabel='True label', xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', rotation_mode='anchor')
        plt.savefig('cm.png')

    @staticmethod
    def lr_scheduler(lr=0.001, multiplier=0.999) -> Generator[float, None, None]:
        """
        Generator of a Learning Rate Scheduler.
        :param lr: Initial Learning Rate.
        :param multiplier: Multiplier factor applier to last iteration LR (0 < multiplier < 1).
        :yields: Descending LR each time next element is called.
        """
        while True:
            yield lr
            lr *= multiplier

    @staticmethod
    def _extract_data(path: str) -> Tuple[List[List[str]], List[str], int]:
        """
        Extracts data from a {Word: POS} file mapping.
        :param path: Path of the data file.
        :return: List of sentences being each one a list of words and each corresponding POS,
        along with length of the longest word.
        """
        max_len_word = 0
        with open(path, mode='r') as f:
            sentences, sentences_pos = [], []
            sentence, sentence_pos = [], []
            for line in f.readlines():
                line = line.strip()
                if line == '':  # Empty line refers to new sentence. Reset.
                    if sentence:
                        sentences.append(sentence)
                        sentences_pos.append(sentence_pos)
                        sentence, sentence_pos = [], []
                else:
                    word, pos = line.split()
                    sentence.append(word)
                    sentence_pos.append(pos)

                    max_len_word = max(max_len_word, len(word))
            if sentence:
                sentences.append(sentence)
        return sentences, sentences_pos, max_len_word

    @staticmethod
    def _store_tensorboard_visualization(path: str):
        """
        Stores a Graph visualization for Tensorboard usage.
        :param path: Path to store the information.
        """
        logging.info('Storing Tensorboard visualization.')
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(path, sess.graph)
            writer.close()


if __name__ == '__main__':
    tf.reset_default_graph()
    logging.getLogger().setLevel(logging.INFO)
    model = WordTagEstimator('train_pos.txt', 'test_pos.txt')
    model.create_graph(emb_dim=100, hid_dim=53, conv_win_size=5)
    model.train(epochs=80, minibatch_size=32, learning_rate=0.0108)
