import os
import shutil
import logging
import numpy as np
import tensorflow as tf

from dataloader import DataLoader, iv3_cache


class FacialRecognition:
    def __init__(self, dataset_path, test_path):
        # Load dataset
        self.img_paths = DataLoader.get_images_in_path(dataset_path)
        self.img_paths_flat = DataLoader.get_flattened_paths(self.img_paths)
        
        # Pretrained InceptionV3
        self.inception_input = tf.get_default_graph().get_tensor_by_name('Model_Input:0')
        self.inception_output = tf.get_default_graph().get_tensor_by_name('Model_Output:0')
        
        # Load test data
        test_data = DataLoader.load_test_data(test_path)
        self.anc_test = self.get_inception_output(test_data[0])
        self.pos_test = self.get_inception_output(test_data[1])
        self.neg_test = self.get_inception_output(test_data[2])
        
        # Hyperparameters
        self.batch_size = 256
        self.patience = 50
        
        # Initialize graph
        self.build_graph()
        
        # Store graph visualization
        self.store_graph()

    def train(self, epochs=500):     
        logging.info('Training start...')
        
        best_accuracy = 0.0
        count_worse = 0
        
        # Annealing Learning Rate
        lr_annealing = self.lr_scheduler()
        
        # Start session and initialize variables
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())        

            for epoch in range(epochs):                
                lr = next(lr_annealing)
                
                for batch_idx, (anc_tensor, pos_tensor, neg_tensor) in enumerate(self.minibatches(self.img_paths)):
                    d = { self.input_anchor: anc_tensor,
                          self.input_positive: pos_tensor,
                          self.input_negative: neg_tensor,
                          self.lr: lr }
                    
                    loss, _ = sess.run([self.loss, self.optimizer], feed_dict=d)
                    
                    msg = 'Minibatch {} processed with loss = {}'
                    logging.debug(msg.format(batch_idx, loss))
                
                d = { self.input_anchor: self.anc_test,
                      self.input_positive: self.pos_test,
                      self.input_negative: self.neg_test }
                    
                epoch_loss, epoch_accuracy = sess.run([self.loss, self.accuracy], feed_dict=d)
                
                # self.saver.save(sess, 'FacialRecognitionModel', global_step=0)
                msg = 'Epoch {:>3}, test acc {:.4f}, test batch loss {:.4f}'
                print(msg.format(epoch, epoch_accuracy, epoch_loss))
                
                # Early stopping
                if epoch_accuracy < best_accuracy:
                    count_worse += 1
                    if count_worse == self.patience:
                        msg = 'Early stopping at epoch {} after {} epochs without test accuracy improvement'
                        print(msg.format(epoch, self.patience))
                        
                        print('Best accuracy: {}%'.format(best_accuracy * 100.0))
                        logging.info('Training finished!')
                        return
                else:
                    count_worse = 0
                    best_accuracy = epoch_accuracy
                    
                    logging.info('Saving model...')
                    self.saver.save(sess, './model.ckpt')
                
        print('Best accuracy: {}%'.format(best_accuracy * 100.0))
        logging.info('Training finished!')

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
            '''Create our network last layers'''
            dense_1 = tf.layers.dense(inp_placeholder,
                                      units=512,
                                      activation='sigmoid',
                                      name='Dense1') 
            dense_2 = tf.layers.dense(dense_1,
                                      units=256,
                                      activation='sigmoid',
                                      activity_regularizer=tf.math.l2_normalize,
                                      name='Dense2')
            dense_3 = tf.layers.dense(dense_2,
                                      units=128,
                                      activation='tanh',
                                      activity_regularizer=tf.math.l2_normalize,
                                      name='Dense3')
            return dense_3
        
        ### Siamese architecture for anchor-positives-negatives
        with tf.variable_scope('siamese', reuse=tf.AUTO_REUSE):
            out_anchor = network(self.input_anchor)
            out_positive = network(self.input_positive)
            out_negative = network(self.input_negative)
        
        ### Loss and optimization
        positive_term = tf.subtract(out_anchor, out_positive)
        positive_term_squared = tf.math.square(positive_term)
        positive_term_norm = tf.reduce_sum(positive_term_squared, axis=1)
        
        negative_term = tf.subtract(out_anchor, out_negative)
        negative_term_squared = tf.math.square(negative_term)
        negative_term_norm = tf.reduce_sum(negative_term_squared, axis=1)
        
        term_loss = tf.subtract(positive_term_norm, negative_term_norm)
        term_loss_margin = tf.add(term_loss, alpha)
        term_loss_margin_positive = tf.maximum(0.0, term_loss_margin)
        self.loss = tf.reduce_mean(term_loss_margin_positive)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        
        ### Evaluation over test set
        sim_positives = tf.math.negative(positive_term_norm)
        corr_positives = tf.math.greater(sim_positives, T)
        corr_positives_mean = tf.reduce_mean(tf.cast(corr_positives, tf.float32))
        
        sim_negatives = tf.math.negative(negative_term_norm)
        corr_negatives = tf.math.less_equal(sim_negatives, T)
        corr_negatives_mean = tf.reduce_mean(tf.cast(corr_negatives, tf.float32))
        
        corr_sum = tf.add(corr_positives_mean, corr_negatives_mean)
        self.accuracy = tf.divide(corr_sum, 2.0)
            
        logging.info('CG built...')
    
    def store_graph(self, folder_name='./tf_summary'):
        '''
        Generate visualization for tensorboard. Run:
        > tensorboard --logdir ./tf_summary/
        '''
        with tf.Session() as sess:
            self.saver = tf.train.Saver()
            
            # Remove summary folder if exists
            if os.path.exists(folder_name):
                logging.debug('Removing old CG visualizations...')
                shutil.rmtree(folder_name, ignore_errors=True)
                
            writer = tf.summary.FileWriter(folder_name, graph=sess.graph)
            writer.close()
            
            logging.info('CG visualization created...')
    
    def minibatches(self, paths, shuffle=True):
        '''
        Geretator to produce minibatches.
        
        :param paths: Paths of the training set images.
        :param batch_size: Size of every minibatch (except maybe for the last one).
        :param shuffle: Shuffle the data the first time we invoke the generator.
        :yields: Pair of triplets (anch, pos, neg) for images and labels respectively.
        '''
        logging.debug('Generating minibatches of size {}...'.format(self.batch_size))
        
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
                
                if current_batch_size == self.batch_size:                    
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

    @iv3_cache
    def get_inception_output(self, paths):
        '''
        Get the output of InceptionV3 for a list of images given their paths.
        
        :param paths: List of images path.
        :returns Tensor of size (?x2048) as output of Inception network.
        '''
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