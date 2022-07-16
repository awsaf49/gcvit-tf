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