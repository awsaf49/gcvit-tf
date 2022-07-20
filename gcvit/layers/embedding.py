import tensorflow as tf

from .feature import ReduceSize


@tf.keras.utils.register_keras_serializable(package="gcvit")
class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.pad = tf.keras.layers.ZeroPadding2D(1, name='pad')
        self.proj = tf.keras.layers.Conv2D(self.dim, kernel_size=3, strides=2, name='proj')
        self.conv_down = ReduceSize(keep_dim=True, name='conv_down')
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.pad(inputs)
        x = self.proj(x)
        x = self.conv_down(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'dim': self.dim})
        return config