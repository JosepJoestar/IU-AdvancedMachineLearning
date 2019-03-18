import os
import shutil
import logging
import tensorflow as tf

N = 5000
CSIZE = 5
LR = 0.001
EPOCHS = 20
NEGATIVES = 20
EMBEDDING_DIM = 50
CONTEXTS_BATCH = 20


class Word2Vec:
    """Word2Vec model (Skip-gram)"""

    def __init__(self, data):
        self.data = data
        self.voc_size = data.voc.num_words

        # Initialize graph
        tf.reset_default_graph()
        self.build_graph()

    def train(self):
        """Train the model with the data and specified Hyperparameters"""
        sess = tf.Session()
        saver = tf.train.Saver()

        if os.path.exists('./tf_summary'):
            logging.debug('Removing old CG visualizations and checkpoints...')
            shutil.rmtree('./tf_summary', ignore_errors=True)

        writer = tf.summary.FileWriter('./tf_summary', graph=sess.graph)

        logging.info('Training started!')

        init = tf.global_variables_initializer()
        sess.run(init)

        step = 0
        global_step = 0
        for epoch in range(1, EPOCHS + 1):
            print('Epoch {}'.format(epoch))
            progress = tf.keras.utils.Progbar(target=len(self.data.voc.text),
                                              stateful_metrics=['batch loss'],
                                              width=40, interval=10)

            batches = self.data.next_batch(CONTEXTS_BATCH, CSIZE, NEGATIVES)
            for last_word_idx, i_central, i_samples, targets in batches:
                step += 1
                d = {self.x_central: i_central,
                     self.x_samples: i_samples,
                     self.y: targets}

                sess.run(self.optimize, feed_dict=d)

                if step % 1000 == 0:
                    _, loss, summ = sess.run([self.embedding, self.loss, self.summary], feed_dict=d)
                    progress.update(last_word_idx, values=[
                        ('batch loss', loss), ('epoch loss', loss)])

                    logging.debug('Saving Summary {}'.format(global_step))
                    writer.add_summary(summ, global_step=global_step)

                    logging.debug('Saving Checkpoint {}'.format(global_step))
                    saver.save(sess, 'tf_summary/model.ckpt', global_step=global_step)

                    global_step += 1

            progress.update(len(self.data.voc.text), values=[
                ('batch loss', loss), ('epoch loss', loss)])

        writer.close()
        sess.close()

    def build_graph(self):
        """Build and return TensorFlow Computational Graph"""
        # Placeholders
        self.x_central = tf.placeholder(tf.int32, shape=[None], name='X_central')
        self.x_samples = tf.placeholder(tf.int32, shape=[None], name='X_samples')
        self.y = tf.placeholder(tf.float32, shape=[None], name='Labels')

        # Weight Matrices
        W1 = tf.Variable(tf.random_uniform(shape=(self.voc_size, EMBEDDING_DIM),
                                           minval=-1, maxval=1, name='W_in'))
        W2 = tf.Variable(tf.random_uniform(shape=(self.voc_size, EMBEDDING_DIM),
                                           minval=-1, maxval=1, name='W_out'))

        # Lookups
        central_lookup = tf.nn.embedding_lookup(W1, self.x_central)
        samples_lookup = tf.nn.embedding_lookup(W2, self.x_samples)

        mult_lookups = tf.multiply(central_lookup, samples_lookup)
        sum_lookups = tf.reduce_sum(mult_lookups, axis=1)

        #  Loss and Optimization

        sigmoid_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,
                                                             logits=sum_lookups)
        self.loss = tf.reduce_mean(sigmoid_ce)
        tf.summary.scalar('loss', self.loss)

        optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=LR)
        self.optimize = optimizer.minimize(self.loss)

        # Final matrix
        embedding_matrix = tf.nn.l2_normalize(0.5 * (W1 + W2), axis=1, name='Normalize')
        self.embedding = tf.Variable(embedding_matrix, name='Embedding')

        self.summary = tf.summary.merge_all()
        logging.info('CG built...')
