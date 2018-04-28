import sys, random, os
from .folder import myImageFolder
import torch

class MultiLabelDataFolder(object):
    def __init__(self, train_dir, label_file, transform_func, train_percent=0.8, seed=0):
        self.__transform_func = transform_func
        self.__label_file = label_file
        self.__train_dir = train_dir
        self.__train_percent = train_percent
        self.__seed = seed        
                
    def _split_data(self, folder, train_percent, allow_caching=True):
        random.seed(self.__seed)
        train_names, val_names = set(), set()

        for filename in os.listdir(folder):
            ddict = train_names if random.random() < train_percent else val_names
            ddict.add(filename)

        return train_names, val_names

    def _get_data_folder(self, names=None):
        transformations = self.__transform_func
        if names is None:
            train_names, val_names = self._split_data(self.__train_dir, self.__train_percent)
        else:
            train_names, val_names = names

        train_folder = myImageFolder(self.__label_file, self.__train_dir, train_names, 
                                                        transform=transformations['train'])
        val_folder = myImageFolder(self.__label_file, self.__train_dir, val_names, 
                                                        transform=transformations['val'])
        if not len(train_folder) or not len(val_folder):
            raise ValueError('One of the image folders contains zero data, train: %s, val: %s' % \
                                (len(train_folder), len(val_folder)))
 
        return {'train':train_folder, 'val':val_folder}