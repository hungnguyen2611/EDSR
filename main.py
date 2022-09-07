import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from preprocess import DataGenerator, dataset_object
from model import EDSR, make_model
from utils import PSNR
import os
import cv2
import glob
import argparse
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import warnings
warnings.filterwarnings("ignore")



def plot_results(lowres, preds):
    """
    Displays low resolution image and super resolution image
    """
    plt.figure(figsize=(24, 14))
    plt.subplot(132), plt.imshow(lowres), plt.title("Low resolution")
    plt.subplot(133), plt.imshow(preds), plt.title("Prediction")
    plt.show()

def train(train_ds, val_ds):
    model = make_model(num_filters=64, num_of_residual_blocks=16)
    opt = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-4, 5e-5]
    ))
    model.compile(optimizer=opt, loss="mae", metrics=[PSNR])
    checkpoint_filepath = 'model/model{epoch:02d}-{val_loss:.2f}.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_PSNR',
        mode='max',
        save_best_only=True)
    model.fit(train_ds, epochs=400, steps_per_epoch=200, validation_data=val_ds, callbacks=[model_checkpoint_callback])


def predict(img_dir, weight_path):
    model = make_model(num_filters=64, num_of_residual_blocks=16)
    model.load_weights(weight_path)
    files = glob.glob(img_dir + '/*')
    for img_file in files:
        highres = cv2.imread(img_file)
        highres = cv2.cvtColor(highres, cv2.COLOR_BGR2RGB)
        lowres = tf.image.random_crop(highres, (250, 250, 3))
        preds = model.predict_step(lowres)
        preds = preds.numpy()
        preds = cv2.cvtColor(preds, cv2.COLOR_RGB2BGR)
        lowres = cv2.cvtColor(lowres.numpy(), cv2.COLOR_RGB2BGR)
        low_res_path = os.path.dirname(img_dir) + '/result/'+ 'low_' + os.path.basename(img_file)
        preds_path = os.path.dirname(img_dir) + '/result/'+ os.path.basename(img_file)
        cv2.imwrite(low_res_path, lowres) 
        print('Saved to {}'.format(low_res_path))
        cv2.imwrite(preds_path, preds) 
        print('Saved to {}'.format(preds_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or predict')
    parser.add_argument('command', metavar='<command>', type = str, help='train or test')
    # parser.add_argument('test', help='test')
    parser.add_argument('--img_dir', required=False, help='image dir to test')
    parser.add_argument('--weight_path', required=False, help='path to model weight')
    args = parser.parse_args()
    if args.command == 'train':
        # download DIV2K from TF datasets
        # using bicubic 4x degradation type
        div2k_data = tfds.image.Div2k(config="bicubic_x4")
        div2k_data.download_and_prepare()
        train = div2k_data.as_dataset(split="train", as_supervised=True)
        train_cache = train.cache()
        val = div2k_data.as_dataset(split="validation", as_supervised=True)
        val_cache = val.cache()
        train_ds = dataset_object(train_cache, training=True)
        val_ds = dataset_object(val_cache, training=False)
        train(train_ds, val_ds)
    if args.command == 'test':
        predict(args.img_dir, args.weight_path)
    
    

    

    

    