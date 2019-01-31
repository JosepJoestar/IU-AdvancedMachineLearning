import tensorflow as tf
from functools import reduce
from numpy import unique, array, vectorize
from sklearn.metrics import accuracy_score, f1_score

class SVMClassifier:

    def __init__(self, train_data=None):
        data, labels = train_data

        labels = self._transform_labels(labels)
        data = self._flatten_input(data)
        
        self.train_data = (data, labels)

        self.assemble_graph()

        self._open_session()

        if train_data:
            self.train()     

    def assemble_graph(self, learning_rate = 0.02):
        raise NotImplementedError()

    def train(self, epochs=20, minibatch_size=256):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()

    def _create_minibatches(self, minibatch_size):
        pos = 0

        data, labels = self.train_data
        n_samples = len(labels)

        batches = []
        while pos + minibatch_size < n_samples:
            batches.append((data[pos:pos+minibatch_size,:], labels[pos:pos+minibatch_size]))
            pos += minibatch_size

        if pos < n_samples:
            batches.append((data[pos:n_samples,:], labels[pos:n_samples,:]))

        return batches

    def _transform_labels(self, labels):
        raise NotImplementedError()
        

    def _flatten_input(self, data):
        raise NotImplementedError()

    def _open_session(self):
        self.sess = tf.Session()





if __name__ == "__main__":



    def mnist_to_binary(train_data, train_label, test_data, test_label):

        binarized_labels = []
        for labels in [train_label, test_label]:
            remainder_2 = vectorize(lambda x: x%2)
            binarized_labels.append(remainder_2(labels))

        train_label, test_label = binarized_labels

        return train_data, train_label, test_data, test_label




    ((train_data, train_labels),
        (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

    train_data, train_labels, test_data, test_labels = mnist_to_binary(train_data, train_labels, eval_data, eval_labels)

    svm = SVMClassifier((train_data, train_labels))
    print("Testing score f1: {}".format(f1_score(test_labels, svm.predict(test_data))))


