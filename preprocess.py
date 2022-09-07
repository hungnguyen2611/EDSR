import numpy as np
import tensorflow as tf
from tensorflow import keras
from augmentation import *
import glob
import os
import cv2

AUTOTUNE = tf.data.experimental.AUTOTUNE

# download DIV2K from TF datasets
# using bicubic 4x degradation type
class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 lr_img_paths,
                 hr_img_paths,
                 batch_size=32,
                 dim=(None, None, 3),
                 shuffle=True,
                 training=False):
        self.dim = dim
        self.batch_size = batch_size
        self.lr_imgs = glob.glob(os.path.join(lr_img_paths, '*.png'))
        self.hr_imgs = glob.glob(os.path.join(hr_img_paths, '*.png'))
        self.shuffle = shuffle
        self.training = training
        self.img_indexes = np.arange(len(self.lr_imgs))
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_indexes) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temps = [self.img_indexes[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temps)
        return X, y
        
    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.lr_imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temps):
        X = []
        y = []
        for i, ID in enumerate(list_IDs_temps):
            lr_img = cv2.cvtColor(cv2.imread(self.lr_imgs[ID]), cv2.COLOR_BGR2RGB).astype(np.float32)
            hr_img = cv2.cvtColor(cv2.imread(self.hr_imgs[ID]), cv2.COLOR_BGR2RGB).astype(np.float32)
            lr_img, hr_img = random_crop(lr_img, hr_img, scale=4)
            if self.training:
                lr_img, hr_img = random_rotate(lr_img, hr_img)
                lr_img, hr_img = flip_left_right(lr_img, hr_img)
            X.append(lr_img)
            y.append(hr_img)
        return np.array(X), np.array(y)

def dataset_object(dataset_cache, batch_size=16, training=True):
    ds = dataset_cache
    ds = ds.map(
        lambda lr, hr: random_crop(lr, hr, scale=4),
        num_parallel_calls=AUTOTUNE
    )
    if training:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)

    # Batching Data
    ds = ds.batch(batch_size)
    if training:
        ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

