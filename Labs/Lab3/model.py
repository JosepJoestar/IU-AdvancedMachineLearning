import logging
import functools
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from dataloader import DataLoader, inceptioncache


def lazyprop(function):
    @property
    @functools.wraps(function)
    def decorator(self):
        attribute = '_cache_' + function.__name__    
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)    
    return decorator


class FacialRecognition:
    def __init__(self, dataset_path):
        
        # Load dataset
        self.img_paths = DataLoader.get_images_in_path(dataset_path)
        
        # Initialize graph
        self.graph
        
        # Annealing Learning Rate
        # lr_annealing = self.lr_scheduler


    def train(self, epochs=500):     
        logging.info('Start training...')
        
        X_train, X_valid = train_test_split(self.img_paths, test_size=0.2)
        
        sess = tf.Session()

        init = tf.global_variables_initializer()
        sess.run(init)
        
        for epoch in range(epochs):
            for batch in self.minibatches(X_train, batch_size=10, shuffle=False):
                anchor_tensor, positive_tensor, negative_tensor = batch
                d = {
                        self.input_anchor: anchor_tensor,
                        self.input_positive: positive_tensor,
                        self.input_negative: negative_tensor,
                        self.lr: 0.0001
                    }
                loss, _ = sess.run([self.loss, self.optimizer], feed_dict=d)
            msg = 'Epoch {}, test acc {:.4f}, test batch loss {:.4f}'
            print(msg.format(epoch, 0, loss))

        sess.close()


    def test(test_data, test_target):
        pass


    @lazyprop
    def graph(self):
        '''Build and return TensorFlow Computational Graph'''
        ### Constants
        alpha = tf.constant(0.2, dtype=tf.float32, name='Alpha')
        T = tf.constant(-0.8, dtype=tf.float32, name='Threshold')   
        
        ### Placeholders
        self.lr = tf.placeholder(dtype=tf.float32, name='LearningRate')
        
        # Inputs for siamese architecture
        self.input_anchor = tf.placeholder(tf.float32, shape=(None, 2048))
        self.input_positive = tf.placeholder(tf.float32, shape=(None, 2048))
        self.input_negative = tf.placeholder(tf.float32, shape=(None, 2048))
        
        ## New Layers
        
        def network(inp_placeholder):        
            dense_1 = tf.layers.dense(inp_placeholder, units=512, activation='sigmoid')            
            dense_2 = tf.layers.dense(dense_1, units=256, activation='sigmoid')
            dense_2_l2_norm = tf.math.l2_normalize(dense_2)
            return tf.layers.dense(dense_2_l2_norm, units=128, activation='tanh')
        
        ### Siamese architecture for anchor-positives-negatives
        with tf.variable_scope('siamese', reuse=tf.AUTO_REUSE):
            out_anchor = network(self.input_anchor)
            out_negative = network(self.input_positive)
            out_positive = network(self.input_negative)
        
        # similarity = tf.math.l2_normalize(tf.subtract(fc, fx))
        # R = tf.subtract(0, similarity)
        
        positive_term = tf.subtract(out_anchor, out_positive)
        positive_term_norm = tf.math.l2_normalize(positive_term)
        
        negative_term = tf.subtract(out_anchor, out_negative)
        negative_term_norm = tf.math.l2_normalize(negative_term)
        
        ith_los = tf.subtract(tf.add(positive_term_norm, alpha), negative_term_norm)
        ith_loss_pos = tf.maximum(0.0, ith_los)
        self.loss = tf.reduce_mean(ith_loss_pos)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
        # Generate visualization for tensorboard
        # > tensorboard --logdir ./tf_summary/
        # with tf.Session() as sess:
            # saver = tf.train.Saver()
            # writer = tf.summary.FileWriter("./tf_summary", graph=sess.graph)
            # writer.close()
    

    def minibatches(self, paths, batch_size=256, shuffle=True):
        '''
        Geretator to produce minibatches.
        
        :param paths: Paths of the training set images.
        :param batch_size: Size of every minibatch (except maybe for the last one).
        :param shuffle: Shuffle the data the first time we invoke the generator.
        :yields: Pair of triplets (anch, pos, neg) for images and labels respectively.
        '''
        if shuffle:
            np.random.shuffle(paths)
        
        n = len(paths)
        random_offset = lambda c, s: (c + 1 + np.random.randint(s - 1)) % s
        
        current_batch_size = 0
        anchor_paths = []
        positive_paths = []
        negative_paths = []
                                      
        for i in range(n):
            m = len(paths[i]['paths'])

            for j in range(m):
                current_batch_size += 1
                
                negative_idx = random_offset(i, n)
                
                # Get triplet paths
                anchor_paths.append(paths[i]['paths'][j])
                positive_paths.append(paths[i]['paths'][random_offset(j, m)])
                negative_paths.append(np.random.choice(paths[negative_idx]['paths']))
                
                # Get triplet labels
                # anchor_label = paths[i]['label']
                # positive_label = paths[i]['label']
                # negative_label = paths[negative_idx]['label']
                
                if current_batch_size == batch_size:
                    anchor_tensor = FacialRecognition.get_inception_output(anchor_paths)
                    positive_tensor = FacialRecognition.get_inception_output(positive_paths)
                    negative_tensor = FacialRecognition.get_inception_output(negative_paths)
                    
                    yield anchor_tensor, positive_tensor, negative_tensor
                    
                    current_batch_size = 0
                    anchor_paths = []
                    positive_paths = []
                    negative_paths = []
        
        if current_batch_size > 0:
            yield anchor_tensor, positive_tensor, negative_tensor
    
    
    @staticmethod
    @inceptioncache
    def get_inception_output(paths):
        inception_input = tf.get_default_graph().get_tensor_by_name('Model_Input:0')
        inception_output = tf.get_default_graph().get_tensor_by_name('Model_Output:0')
        
        img = DataLoader.images_to_np(paths)
        with tf.Session() as sess:
            d = { inception_input: np.reshape(img, (-1, 299, 299, 3)) }
            img_repr = sess.run(inception_output, feed_dict=d)

        return img_repr


    @lazyprop
    def lr_scheduler(self, lr=0.0001, multiplier=0.999):
        '''
        Generator of a Learning Rate Scheduler.
        :param lr: Initial Learning Rate.
        :param multiplier: Multiplier factor applier to last iteration LR (0 < multiplier < 1).
        :yields: Descending LR each time next element is called.
        '''
        while True:
            yield lr
            lr *= multiplier