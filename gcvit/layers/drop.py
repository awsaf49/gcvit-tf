import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="gcvit")
class Identity(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return tf.identity(x)

    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable(package="gcvit")
class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=0., scale_by_keep=True, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, x, training=None):
        if self.drop_prob==0. or not training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
        random_tensor = tf.floor(random_tensor)
        if keep_prob > 0.0 and self.scale_by_keep:
            x = (x / keep_prob) 
        return x * random_tensor

    def get_config(self):
        config = super().get_config()
        config.update({
            "drop_prob": self.drop_prob,
            "scale_by_keep": self.scale_by_keep
            })
        return config