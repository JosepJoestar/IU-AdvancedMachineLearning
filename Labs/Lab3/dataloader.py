import os
import cv2
import pickle
import logging
import numpy as np
import tensorflow as tf


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
        cache = InceptionCache()
        images = cache.get(args[1])
        
        missing_indices = []
        missing_paths = []
        
        for i in range(len(images)):
            if images[i] is None:
                missing_indices.append(i)
                missing_paths.append(args[1][i])
       
        cached_images = len(images) - len(missing_paths)
        logging.debug('{}/{} images cached'.format(cached_images, len(images)))
        
        if len(missing_paths) > 0:
            missing_images = f(args[0], missing_paths)
            for i in range(len(missing_paths)):
                cache.put(missing_paths[i], missing_images[i])
                images[missing_indices[i]] = missing_images[i]
                
            cache.dump()
                
        return np.stack(images)
    return wrapper


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class InceptionCache(metaclass=Singleton):
    '''Cache that stores representation after Inception NNET (?, 2048)'''
    
    PKL_FILE = 'inception_cached.pkl'
    
    def __init__(self):
        '''Initialize cache with previous data or new want if not existing'''
        if os.path.exists(InceptionCache.PKL_FILE):
            self.cache = pickle.load(open(InceptionCache.PKL_FILE, mode='rb'))
            logging.info('Loaded cached data ({} elements)'.format(len(self.cache)))
        else:
            self.cache = {}
        
    def get(self, paths):
        '''
        Get a list of cached elements from their paths
        
        :param paths: List of paths to check in cache
        :returns: List of cached elements (with None in non-existing elements)
        '''
        f = lambda p: self.cache[p] if p in self.cache else None
        return list(map(f, paths))

    def put(self, path, element):
        '''
        Add an element into the cache
        
        :param path: Key of the new element
        :param element: Output to cache
        '''
        self.cache[path] = element
        
    def dump(self):
        '''Dump cache into a file for later executions usage'''
        pickle.dump(self.cache, open(InceptionCache.PKL_FILE, mode='wb'))


class DataLoader:
    '''Helper class to load image data into Numpy format'''
        
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

        return list_images_path
    
    @staticmethod
    def get_flattened_paths(struct):
        '''
        Transforms dataset folder structure into a flattened list
        
        :param struct: Folder struct returned by #get_images_in_path
        :returns: List of tuples (path, label, idx_folder, idx_path_in_folder)
        '''
        flat = []
        for i, folder in enumerate(struct):
            for j, path in enumerate(folder['paths']):
                tup = (path, folder['label'], i, j)
                flat.append(tup)
                
        return flat

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
        :returns: Tensor of images of shape (?, 299, 299, 3).
        '''
        load_image = lambda p: cv2.resize(cv2.imread(p), (299, 299))
        return np.stack(list(map(load_image, paths)))
