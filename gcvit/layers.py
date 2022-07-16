import tensorflow as tf
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
        outputs = self.fc1(inputs)
        outputs = self.act(outputs)
        outputs = self.drop(outputs)
        outputs = self.fc2(outputs)
        outputs = self.drop(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_features":self.hidden_features, 
            "out_features":self.out_features, 
            "act_layer":self.act_layer,
            "drop":self.drop_rate
            })
        return config