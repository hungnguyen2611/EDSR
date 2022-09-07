import numpy as np
import tensorflow as tf

def flip_left_right(lr, hr):
    """Flips images to left or right randomly"""
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(
        rn < 0.5,
        lambda: (lr, hr),
        lambda: (tf.image.flip_left_right(lr), tf.image.flip_left_right(hr))
    )
    

def random_rotate(lr, hr):
    """rotate image by 90 degrees randomly"""
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr, rn), tf.image.rot90(hr, rn)


def random_crop(lr_img, hr_img, hr_crop_size=96, scale=4):
    """Crop images.
    sub-image for larger training data-set
    low resolution images: 24x24
    high resolution images: 96x96
    """
    lr_crop_size = hr_crop_size // scale  # 96//4=24
    lr_img_shape = tf.shape(lr_img)[:2]  # (height,width)

    lr_width = tf.random.uniform(
        shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32
    )
    lr_height = tf.random.uniform(
        shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32
    )

    hr_width = lr_width * scale
    hr_height = lr_height * scale

    lr_img_cropped = lr_img[
        lr_height : lr_height + lr_crop_size,
        lr_width : lr_width + lr_crop_size,
    ]  # 24x24
    hr_img_cropped = hr_img[
        hr_height : hr_height + hr_crop_size,
        hr_width : hr_width + hr_crop_size,
    ]  # 96x96

    return lr_img_cropped, hr_img_cropped