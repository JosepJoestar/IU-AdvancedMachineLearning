import tensorflow as tf

from nn import NN


class FNN(NN):
    def create_graph(self, input_dim, vocab_dim, emb_dim=5, lr=0.001, unit_layers=[5], activations=['relu'],
                     mode=('flatten', 1)):
        # Word embeddings
        embedded_names = self._create_graph_input(input_dim, vocab_dim, emb_dim)
        flatten_names = self._flatten_input(embedded_names, input_dim, emb_dim, mode)

        #  Add Dense Layers
        layers = [flatten_names]
        for idx, units in enumerate(unit_layers):
            layers.append(tf.layers.dense(layers[-1], units, activation=activations[idx]))

        last_layer = layers[-1]
        self._create_graph_output(last_layer, lr)

    def _flatten_input(self, tensor, input_dim, emb_dim, mode):
        assert len(mode) == 2
        assert mode[0] in ['flatten', 'max_pool', 'average', 'w_average']

        if mode[0] == 'max_pool':
            return tf.reduce_max(tensor, axis=mode[1])
        elif mode[0] == 'average':
            return tf.reduce_mean(tensor, axis=mode[1])
        elif mode == 'w_average':
            filt = tf.get_variable('filter', shape=(1, input_dim, 1) if mode[1] == 1 else (1, 1, emb_dim))
            return tf.reduce_mean(tensor * filt, axis=mode[1])
        else:
            return tf.reshape(tensor, shape=(-1, input_dim * emb_dim))
