import logging
import numpy as np
import tensorflow as tf

# from sklearn.model_selection import train_test_split

from dataloader import DataLoader, inceptioncache


class FacialRecognition:
    def __init__(self, dataset_path):
        # Load dataset
        self.img_paths = DataLoader.get_images_in_path(dataset_path)
        self.img_paths_flat = DataLoader.get_flattened_paths(self.img_paths)
        
        # Pretrained InceptionV3
        self.inception_input = tf.get_default_graph().get_tensor_by_name('Model_Input:0')
        self.inception_output = tf.get_default_graph().get_tensor_by_name('Model_Output:0')
        
        # Initialize graph
        self.build_graph()

    def train(self, epochs=500):     
        logging.info('Training start...')
        
        # Annealing Learning Rate
        lr_annealing = self.lr_scheduler()
    
        # Split dataset into training and test
        # X_train_flat, X_valid_flat = train_test_split(self.img_paths_flat, test_size=0.2)
        X_train = self.img_paths
        
        # Start session and initialize variables
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            lr = next(lr_annealing)
            
            epoch_loss = 0
            epoch_accuracy = 0
            
            for batch_idx, (anc_tensor, pos_tensor, neg_tensor) in enumerate(self.minibatches(X_train)):
                d = { self.input_anchor: anc_tensor,
                      self.input_positive: pos_tensor,
                      self.input_negative: neg_tensor,
                      self.lr: lr }
                
                loss, _ = sess.run([self.loss, self.optimizer], feed_dict=d)
                
                msg = 'Minibatch {} processed with loss = {} and accuracy {}'
                logging.debug(msg.format(batch_idx, loss, -1))
                
                epoch_loss += loss
                # epoch_accuracy += acc
                
            # Epoch accuracy and loss by mean of each batch
            epoch_loss /= (batch_idx + 1)
            epoch_accuracy /= (batch_idx + 1)
            
            # self.saver.save(sess, 'FacialRecognitionModel', global_step=0)
            msg = 'Epoch {:>3}, test acc {:.4f}, test batch loss {:.4f}'
            print(msg.format(epoch, -1, epoch_loss))

        # Close session after training is finished
        sess.close()
        
        logging.info('Training finished!')

    def test(test_data, test_target):
        pass

    def build_graph(self):
        '''Build and return TensorFlow Computational Graph'''
        logging.info('Building TF-CG...')
        
        ### Constants
        alpha = tf.constant(0.2, dtype=tf.float32, name='Alpha')
        T = tf.constant(-0.8, dtype=tf.float32, name='Threshold')   
        
        ### Placeholders
        self.lr = tf.placeholder(dtype=tf.float32, name='LearningRate')
        self.input_anchor = tf.placeholder(tf.float32, shape=(None, 2048), name='Anchors')
        self.input_positive = tf.placeholder(tf.float32, shape=(None, 2048), name='Positives')
        self.input_negative = tf.placeholder(tf.float32, shape=(None, 2048), name='Negatives')
        
        ## New Layers
        def network(inp_placeholder):        
            dense_1 = tf.layers.dense(inp_placeholder, units=512, activation='sigmoid', name='Dense1')            
            dense_2 = tf.layers.dense(dense_1, units=256, activation='sigmoid', name='Dense2')
            dense_2_l2_norm = tf.math.l2_normalize(dense_2, name='Dense2_L2')
            return tf.layers.dense(dense_2_l2_norm, units=128, activation='tanh', name='Dense3')
        
        ### Siamese architecture for anchor-positives-negatives
        with tf.variable_scope('face_feature_extraction', reuse=tf.AUTO_REUSE):
            out_anchor = network(self.input_anchor)
            out_negative = network(self.input_positive)
            out_positive = network(self.input_negative)
        
        positive_term = tf.subtract(out_anchor, out_positive)
        positive_term_norm = tf.math.l2_normalize(positive_term)
        
        negative_term = tf.subtract(out_anchor, out_negative)
        negative_term_norm = tf.math.l2_normalize(negative_term)
        
        ith_los = tf.subtract(tf.add(positive_term_norm, alpha), negative_term_norm)
        ith_loss_pos = tf.maximum(0.0, ith_los)
        self.loss = tf.reduce_mean(ith_loss_pos)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
        # Similarity function
        # def similarity(a, b):
        #     dif = tf.subtract(a, b)
        #     l2_dif = tf.math.l2_normalize(dif)
        #     return tf.subtract(0.0, l2_dif)
        
        # sim_pos = similarity(out_anchor, out_positive)
        # corr_pos = tf.math.greater(sim_pos, T)
        # corr_pos_count = tf.reduce_sum(tf.cast(corr_pos, tf.float32))
        
        # sim_neg = similarity(out_anchor, out_negative)
        # corr_neg = tf.math.less_equal(sim_neg, T)
        # corr_neg_count = tf.reduce_sum(tf.cast(corr_neg, tf.float32))
        
        # corrects = tf.add(corr_pos_count, corr_neg_count)
        # self.accuracy = tf.divide(corrects, 256.0 * 2)
        
        # Generate visualization for tensorboard
        # > tensorboard --logdir ./tf_summary/
        with tf.Session() as sess:
            self.saver = tf.train.Saver()
            writer = tf.summary.FileWriter("./tf_summary", graph=sess.graph)
            writer.close()
            
        logging.info('CG builded...')
    
    def minibatches(self, paths, batch_size=256, shuffle=True):
        '''
        Geretator to produce minibatches.
        
        :param paths: Paths of the training set images.
        :param batch_size: Size of every minibatch (except maybe for the last one).
        :param shuffle: Shuffle the data the first time we invoke the generator.
        :yields: Pair of triplets (anch, pos, neg) for images and labels respectively.
        '''
        logging.debug('Generating minibatches of size {}...'.format(batch_size))
        
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
                    anchor_tensor = self.get_inception_output(anchor_paths)
                    positive_tensor = self.get_inception_output(positive_paths)
                    negative_tensor = self.get_inception_output(negative_paths)
                    
                    logging.debug('Minibatch generated!')
                    yield anchor_tensor, positive_tensor, negative_tensor
                    
                    current_batch_size = 0
                    anchor_paths = []
                    positive_paths = []
                    negative_paths = []
        
        if current_batch_size > 0:
            anchor_tensor = self.get_inception_output(anchor_paths)
            positive_tensor = self.get_inception_output(positive_paths)
            negative_tensor = self.get_inception_output(negative_paths)
            
            logging.debug('Last Minibatch generated!')
            yield anchor_tensor, positive_tensor, negative_tensor
    
    @inceptioncache
    def get_inception_output(self, paths):
        imgs = DataLoader.images_to_np(paths)
        
        with tf.Session() as sess:
            d = { self.inception_input: imgs }
            img_repr = sess.run(self.inception_output, feed_dict=d)

        return img_repr

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