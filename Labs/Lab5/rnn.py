import tensorflow as tf

from nn import NN


class RNN(NN):
    def create_graph(self, input_dim, vocab_dim, emb_dim=5, lr=0.001,
                     unit_layers=[5], activations=['relu'], cell_type=[('LSTM', False)]):
        assert len(unit_layers) > 0
        assert len(unit_layers) == len(activations)

        # Word embeddings
        embedded_names = self._create_graph_input(input_dim, vocab_dim, emb_dim)

        #  Add Layers
        cell_layers = []
        for idx, hidden_dim in enumerate(unit_layers):
            if cell_type[idx][0] == 'LSTM':
                cell_layers.append(tf.nn.rnn_cell.LSTMCell(hidden_dim, activation=activations[idx],
                                                           use_peepholes=cell_type[idx][1]))
            else:
                cell_layers.append(tf.nn.rnn_cell.GRUCell(hidden_dim, activation=activations[idx]))

        #  Multilayer cell
        cell = tf.contrib.rnn.MultiRNNCell(cell_layers, state_is_tuple=True)

        #  Dynamic RNN (Dynamic graph with a loop) using defined cell
        outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=embedded_names, dtype=tf.float32)

        #  Filter only last timestep output
        last_layer = outputs[:, -1, :]
        self._create_graph_output(last_layer, lr)
