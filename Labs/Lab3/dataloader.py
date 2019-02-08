import os
import cv2
import logging
import numpy as np
import tensorflow as tf

from functools import reduce

ISIZE = 299 # CNN model required image size


def load_inception_graph(pretrained_graph_path):
    '''
    Input Tensor -> Model_Input(None, 299, 299, 3)
    Output Tensor -> Model_Output(None, 2048)
    '''
    with tf.Session() as sess:
        with tf.gfile.GFile(pretrained_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            
        return 'Model_Input:0', 'Model_Output:0'
    
    

def inceptioncache(f):
    '''Decorator fot catching data output after InceptionV3 network'''
    def wrapper(*args, **kwargs):
        images = InceptionCache.get(args[0])
        
        missing_indices = []
        missing_paths = []
        
        for i in range(len(images)):
            if images[i] is None:
                missing_indices.append(i)
                missing_paths.append(args[0][i])
       
        cached_images = len(images) - len(missing_indices)
        logging.info('{}/{} images cached'.format(cached_images, len(images)))
        
        if len(missing_paths) > 0:
            missing_images = f(missing_paths)
            for i in range(len(missing_paths)):
                InceptionCache.put(missing_paths[i], missing_images[i])
                images[missing_indices[i]] = missing_images[i]
                
        return np.stack(images)
    return wrapper


class InceptionCache:
    '''Cache that stores representation after Inception NNET (?, 2048)'''
    cache = {}

    @staticmethod
    def get(paths):
        f = lambda p: InceptionCache.cache[p] if p in InceptionCache.cache else None
        return list(map(f, paths))

    @staticmethod
    def put(path, element):
        InceptionCache.cache[path] = element


class DataLoader:
    '''
    Helper class to load image data into Numpy format.
    '''
        
    @staticmethod
    def get_images_in_path(path, filter=True):
        '''
        Iterates over the dataset folder building a list with all the images.
        
        :param path: System path from where images will be loaded.
        :param filter: Set True to filter solitary images (training purposes).
        :returns: List[ Dict{ label: str, paths: List[str] } ].
        '''
        if not os.path.isdir(path):
            msg = 'ERROR: Specified path "{}" does not exist or is not a directory'
            print(msg.format(path))
            exit(1)

        list_images_path = []
        for dirname, _, filenames in os.walk(path):
            # Filter folders with a single image
            if filter and len(filenames) <= 1:
                continue
            
            # Add next category label
            label = dirname.split('/')[1]
            list_images_path.append({ 'label': label, 'paths': []})
            
            # Add next category images
            for filename in filenames:
                file_path = dirname + '/' + filename
                list_images_path[-1]['paths'].append(file_path)

        return list_images_path[:50]
    
    @staticmethod
    def get_flattened_paths(struct):
        return reduce(lambda acc, x: acc + x['paths'], struct, [])

    @staticmethod
    def get_labels_by_paths(paths):
        '''
        Maps images paths do category labels.
        
        :param paths: List of images paths.
        :returns: List of corresponding labels.
        '''
        path_to_label = lambda p: p.split('/')[1]
        return list(map(path_to_label, paths))
    
    @staticmethod
    def images_to_np(paths):
        '''
        Takes the list of images path in the dataset and retuns the list of images.
        This images are shaped as (?, ISIZE, ISIZE, 3).
        The first dimension size is the number of images loaded.
        
        :param labels: Returns
        :returns: Tensor of images of shape (?, ISIZE, ISIZE, 3).
        '''
        load_image = lambda p: cv2.resize(cv2.imread(p), (ISIZE, ISIZE))
        return list(map(load_image, paths))
