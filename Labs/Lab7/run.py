import os
import argparse
from functools import reduce
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf

from Labs.Lab7.utils import load_data, create_dirs, create_minibatches, get_current_time
from Labs.Lab7.utils import write_to_tensorboard, create_summary_and_projector, create_evaluation_tensor

# This project uses tensorboard. You can launch tensorboard by executing
# $ tensorboard --logdir log --port 6006
# in your project folder.

# Set parameters
LOG_DIR = 'log'
EPOCHS = 50


def create_model(input_shape, params: Dict):
    """
    Create a simple autoencoder model. Input is assumed to be an image
    :param params: Network hyperparameters ƒrom random search
    :param input_shape: expects the input in format (height, width, n_channels)
    :return: dictionary with tensors required to train and evaluate the model
    """
    h, w, c = input_shape

    # Encoder
    inp = tf.placeholder(shape=[None, h, w, c], dtype=tf.float32)
    l_enc = tf.layers.flatten(inp)

    for (neurons, activation) in params['layers_enc']:
        l_enc = tf.layers.dense(l_enc, units=neurons, activation=activation)

    # Latent Space
    encoding = tf.layers.dense(l_enc, units=params['latent_space_size'], activation=None, name='encoded')

    # Decoder
    l_dec = encoding
    for (neurons, activation) in params['layers_dec']:
        l_dec = tf.layers.dense(l_dec, units=neurons, activation=activation)
    l_dec = tf.layers.dense(l_dec, h * w * c, activation='tanh')

    # any layer without activation could be named as logits
    logits = tf.reshape(l_dec, [-1, h, w, c], name='logits')
    decode = tf.nn.sigmoid(logits, name='decoded')
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=inp, logits=logits, name='loss')
    cost = tf.reduce_mean(loss, name='cost')

    optimizer = tf.train.AdamOptimizer(params['learning_rate']).minimize(cost)

    # -- END CODE HERE --

    model = {'cost': cost,
             'input': inp,
             'enc': encoding,
             'opt': optimizer,
             'dec': decode}
    return model


def random_search(n: int):
    def generate_random_dec_sequence(m: int, dim: int, desc: bool) -> List[int]:
        seq = np.arange(start=dim, stop=28 * 28, step=5)
        seq = sorted(np.random.permutation(seq)[:m], reverse=desc)
        return seq

    def generate_layers(dim: int, desc=True) -> List[Tuple[int, str]]:
        num_layers = np.random.randint(2, 6)
        activations = np.random.choice(['tanh', 'relu', 'elu'], num_layers)
        sizes = generate_random_dec_sequence(num_layers, dim, desc)
        if not desc:
            sizes = sizes[::-1]
        return list(zip(sizes, activations))

    latent_dim = 10 * np.random.randint(low=1, high=31)  # [10, 20, ..., 300]

    for i in range(n):
        yield {
            'layers_enc': generate_layers(latent_dim, desc=True),
            'layers_dec': generate_layers(latent_dim, desc=False),
            'batch_size': 2 ** np.random.randint(low=4, high=9),  # [32, 64, ..., 512]
            'latent_space_size': latent_dim,
            'learning_rate': np.random.uniform(low=0.0001, high=0.005)  # Uniform in [0.0001, 0.005]
        }


def train_model(sess: tf.Session, params: Dict):
    # Create necessary directories
    current_run = get_current_time()
    log_path, run_path = create_dirs(LOG_DIR, current_run)

    # Load MNIST data
    images, labels = load_data()
    mbs = create_minibatches(images, labels, params['batch_size'])

    # Prepare evaluation set
    # this set is used to visualize embedding space and decoding results
    evaluation_set = reduce(lambda acc, x: acc + list(x), mbs[0:(512 // params['batch_size'])], [])
    evaluation_shape = (params['batch_size'], params['latent_space_size'])

    print('Initializing model')
    input_shape = (28, 28, 1)
    model = create_model(input_shape, params)
    evaluation = create_evaluation_tensor(model, evaluation_shape)
    sess.run(tf.global_variables_initializer())

    for_tensorboard = create_summary_and_projector(model, evaluation, evaluation_set, run_path)
    train_writer = tf.summary.FileWriter(run_path, sess.graph)
    saver = tf.train.Saver()

    for e in range(EPOCHS):
        # iterate through minibatches
        epoch_cost = 0
        for mb in mbs:
            batch_cost, _ = sess.run([model['cost'], model['opt']],
                                     feed_dict={model['input']: mb[0]})

            epoch_cost += (batch_cost / len(mbs))

        # write current results to log
        write_to_tensorboard(sess, train_writer, for_tensorboard, evaluation_set, evaluation, e)
        # save trained model
        saver.save(sess, os.path.join(run_path, 'model.ckpt'))

        print('\tEpoch: {}/{}'.format(e + 1, EPOCHS), 'batch cost: {:.4f}'.format(epoch_cost))


def main(tuning=False):
    if tuning:
        for param_config in random_search(60):
            tf.reset_default_graph()
            tf.set_random_seed(1)

            with tf.Session() as sess:
                train_model(sess, param_config)
    else:
        tf.reset_default_graph()
        config = {
            'layers_enc': [(460, 'elu'), (350, 'tanh')],
            'layers_dec': [(450, 'elu'), (500, 'tanh')],
            'batch_size': 128,
            'latent_space_size': 250,
            'learning_rate': 0.00085995460828
        }
        with tf.Session() as sess:
            train_model(sess, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tune', default=False)
    args = parser.parse_args()

    main(args.tune)
