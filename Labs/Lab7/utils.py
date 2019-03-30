import gzip
import numpy as np
import os
from datetime import datetime
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import pandas as p

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def normalize(imgs):
    """
    Transform inputs so that pixel values are in range 0 to 1
    :param imgs: numpy array with images
    :return: normalized numpy array
    """
    min_val = np.min(imgs)
    max_val = np.max(imgs)
    imgs_n = (imgs - min_val) / (max_val - min_val)
    return imgs_n


def load_data():
    imgs = extract_data("./MNIST_data/train-images-idx3-ubyte.gz", 60000)
    lbls = extract_labels("./MNIST_data/train-labels-idx1-ubyte.gz", 60000)
    return normalize(imgs), lbls


def create_dirs(log_dir, current_run):
    current_path = os.getcwd()
    log_path = os.path.join(current_path, log_dir)
    print(log_path)
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    run_path = os.path.join(log_path, current_run)
    if not os.path.isdir(run_path):
        os.mkdir(run_path)
    return log_path, run_path


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def create_minibatches(data, labels, mb_size):
    """

    :param data: numpy array of shape (n_samples,...) where the first dimension iterates through samples
    :param labels: numpy array with labels
    :param mb_size: samples per minibatch
    :return: list of tuples (minibatch_data, minibatch_labels)
    """
    n_samples = data.shape[0]
    full_mb = n_samples // mb_size

    mbs = []

    for i in range(full_mb):
        mb = data[i * mb_size: (i + 1) * mb_size, ...]
        lbl = labels[i * mb_size: (i + 1) * mb_size, ...]
        mbs.append((mb, lbl))

    if n_samples > full_mb * mb_size:
        mb = data[full_mb * mb_size:]
        lbl = labels[full_mb * mb_size:]
        mbs.append((mb, lbl))

    return mbs


def write_to_tensorboard(sess, writer, for_tensorboard, evaluation_set, evaluation, epoch_index):
    """
    save the results of evaluation to tensorboard
    :param sess: current session
    :param writer: tensorflow writer object
    :param for_tensorboard: dictionary with tensors for image and scalar summary generation
    :param evaluation_set: tuple that stores the data used for evaluation and corresponding labels.
    Shape should be the same as for evaluation tensors
    :param evaluation: dictionary that holds evaluation tensor, variable for storing embeddings, and input placeholder
    :param epoch_index:
    :return: Nothing
    """
    summ, _ = sess.run([for_tensorboard['summary'], evaluation['assign_embedding']],
                       feed_dict={evaluation['input']: evaluation_set[0]})
    writer.add_summary(summ, epoch_index)
    projector.visualize_embeddings(writer, for_tensorboard['projector'])


def create_summary_and_projector(model, evaluation, evaluation_set, run_path, max_to_output=10):
    """
    Set up objects to store embeddings and learning loss for the use in tensorboard
    :param model: dictionary that holds model tensors
    :param evaluation: dictionary that holds evaluation tensors
    :param evaluation_set: tuple that stores the data used for evaluation and corresponding labels.
    Shape should be the same as for evaluation tensors.
    :param run_path: path to current project
    :param max_to_output: number of sample images to output into tensorboard
    :return: dictionary with summary generation and embedding projector tensors
    """

    def store_labels(mb):
        emb_lbls = mb[1].reshape(-1, 1)
        p.DataFrame(emb_lbls).to_csv(os.path.join(run_path, "emb_lbls.tsv"), index=False, header=False, sep='\t')

    # https: // www.tensorflow.org / programmers_guide / summaries_and_tensorboard
    summ_sc = tf.summary.scalar(name="training_loss", tensor=model['cost'])
    summ_im_inp = tf.summary.image(name="original_image", tensor=model['input'], max_outputs=max_to_output)
    summ_im_dec = tf.summary.image(name="decoded_image", tensor=model['dec'], max_outputs=max_to_output)
    summ = tf.summary.merge([summ_sc, summ_im_inp, summ_im_dec])

    # https: // www.tensorflow.org / programmers_guide / embedding
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = evaluation['embedding_variable'].name
    embedding.metadata_path = "emb_lbls.tsv"

    store_labels(evaluation_set)

    return {'summary': summ, 'projector': config}


def create_evaluation_tensor(model, evaluation_shape):
    """
    Create tensors that are used for visualizing in tensorboard
    :param model: dictionary that holds model tensors
    :param evaluation_shape: tuple (n_samples_for_evaluation, size_of_hidden_space)
    :return: dictionary that holds tensors for computing image embeddings
    """
    # prepare variable for visualizing embeddings
    # beware of fixed input size
    # ev = tf.placeholder(shape=evaluation_shape, dtype=tf.float32, name="emb_placeholder")
    embedding_var = tf.Variable(np.zeros(evaluation_shape), dtype=tf.float32, name="embeddings")
    emb_assign = tf.assign(embedding_var, model['enc'])

    evaluation = {'assign_embedding': emb_assign,
                  'embedding_variable': embedding_var,
                  'input': model['input']}
    return evaluation
