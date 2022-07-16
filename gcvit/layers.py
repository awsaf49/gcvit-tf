import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras


@tf.keras.utils.register_keras_serializable(package="gcvit")
class MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_features=None, out_features=None, act_layer='gelu', drop=0., **kwargs):
        super().__init__(**kwargs)
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.drop_rate = drop

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.hidden_features = self.hidden_features or self.in_features
        self.out_features = self.out_features or self.in_features
        self.fc1 = tf.keras.layers.Dense(self.hidden_features, name="fc1")
        self.act = tf.keras.layers.Activation(self.act_layer, name="act")
        self.fc2 = tf.keras.layers.Dense(self.out_features, name="fc2")
        self.drop = tf.keras.layers.Dropout(self.drop_rate, name="drop")  # won't show up in tf.keras.Model summary
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_features":self.hidden_features, 
            "out_features":self.out_features, 
            "act_layer":self.act_layer,
            "drop":self.drop_rate
            })
        return config


@tf.keras.utils.register_keras_serializable(package="gcvit")
class SE(tf.keras.layers.Layer):
    def __init__(self, oup=None, expansion=0.25, **kwargs):
        super().__init__(**kwargs)
        self.expansion = expansion
        self.oup = oup

    def build(self, input_shape):
        inp = input_shape[-1]
        self.oup = self.oup or inp
        self.avg_pool = tfa.layers.AdaptiveAveragePooling2D(1, name="avg_pool")
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(int(inp * self.expansion), use_bias=False, name='fc/0'),
            tf.keras.layers.Activation('gelu', name='fc/1'),
            tf.keras.layers.Dense(self.oup, use_bias=False, name='fc/2'),
            tf.keras.layers.Activation('sigmoid', name='fc/3')
        ])
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        b, _, _, c = tf.shape(inputs)
        x = tf.reshape(self.avg_pool(inputs), (b, c))
        x = tf.reshape(self.fc(x), (b, 1, 1, c))
        return x*inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'expansion': self.expansion,
            'oup': self.oup,
            })
        return config


@tf.keras.utils.register_keras_serializable(package="gcvit")
class ReduceSize(tf.keras.layers.Layer):
    def __init__(self, keep_dim=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_dim = keep_dim

    def build(self, input_shape):
        dim = input_shape[-1]
        dim_out = dim if self.keep_dim else 2*dim
        self.pad = tf.keras.layers.ZeroPadding2D(1, name='pad')
        self.conv = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='valid', use_bias=False, name='conv/0'),
            tf.keras.layers.Activation('gelu', name='conv/1'),
            SE(name='conv/2'),
            tf.keras.layers.Conv2D(dim, kernel_size=1, strides=1, padding='valid', use_bias=False, name='conv/3')
        ])
        self.reduction = tf.keras.layers.Conv2D(dim_out, kernel_size=3, strides=2, padding='valid', use_bias=False,
                                                name='reduction')
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm1')  # eps like PyTorch
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm2')
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.norm1(inputs)
        x = x + self.conv(self.pad(x))  # if pad had weights it would've thrown error with .save_weights()
        x = self.reduction(self.pad(x))
        x = self.norm2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "keep_dim":self.keep_dim,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="gcvit")
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.pad = tf.keras.layers.ZeroPadding2D(1)
        self.proj = tf.keras.layers.Conv2D(self.dim, kernel_size=3, strides=2, name='proj')
        self.conv_down = ReduceSize(keep_dim=True, name='conv_down')
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.proj(self.pad(inputs))
        x = self.conv_down(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'dim': self.dim})
        return config


@tf.keras.utils.register_keras_serializable(package="gcvit")
class FeatExtract(tf.keras.layers.Layer):
    def __init__(self, keep_dim=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_dim = keep_dim

    def build(self, input_shape):
        dim = input_shape[-1]
        self.pad = tf.keras.layers.ZeroPadding2D(1, name='pad')
        self.conv = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='valid', use_bias=False, name='conv/0'),
            tf.keras.layers.Activation('gelu', name='conv/1'),
            SE(name='conv/2'),
            tf.keras.layers.Conv2D(dim, kernel_size=1, strides=1, padding='valid', use_bias=False, name='conv/3')
        ])
        if not self.keep_dim:
            self.pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', name='pool')
        # else:
        #     self.pool = tf.keras.layers.Activation('linear', name='identity')  # hack for PyTorch nn.Identity layer ;)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs + self.conv(self.pad(inputs))  # if pad had weights it would've thrown error with .save_weights()
        if not self.keep_dim:
            x = self.pool(self.pad(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "keep_dim":self.keep_dim,
        })
        return config