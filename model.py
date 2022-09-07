import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np 
class EDSR(tf.keras.Model):

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # update metric
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, x):
        x = tf.cast(tf.expand_dims(x, axis=0), tf.float32)
        super_resolution_img = self(x, training=False)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        super_resolution_img = tf.round(super_resolution_img)
        super_resolution_img = tf.squeeze(
            tf.cast(super_resolution_img, tf.uint8), axis=0
        )
        return super_resolution_img


# defining some blocks
# residual block
def res_block(inputs):
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.Add()([inputs, x])
    return x

# up-sampling block
def upsampling_block(inputs, factor=2, **kwargs):
    x = layers.Conv2D(64 * (factor ** 2), 3, padding='same', **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor) # sub-pixel convolution, x2 up-sampling
    x = layers.Conv2D(64 * (factor ** 2), 3, padding='same')(x)
    x = tf.nn.depth_to_space(x, block_size=factor)
    return x


def make_model(num_filters, num_of_residual_blocks):
    input_layer = layers.Input(shape=(None, None, 3), dtype=np.float32)
    x = tf.cast(input_layer, tf.float32)/255
    x = x_new = keras.layers.Conv2D(num_filters, 3, padding='same')(x)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks):
        x_new = res_block(x_new)

    x_new = layers.Conv2D(num_filters, 3, padding='same')(x_new)
    x = layers.Add()([x, x_new])
    x = upsampling_block(x)
    x = layers.Conv2D(3, 3, padding='same')(x)

    output_layer = x*255
    return EDSR(input_layer, output_layer)