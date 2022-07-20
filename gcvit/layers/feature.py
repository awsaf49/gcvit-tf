import tensorflow as tf
import tensorflow_addons as tfa

H_AXIS = -3
W_AXIS = -2

@tf.keras.utils.register_keras_serializable(package="gcvit")
class Mlp(tf.keras.layers.Layer):
    def __init__(self, hidden_features=None, out_features=None, act_layer='gelu', dropout=0., **kwargs):
        super().__init__(**kwargs)
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.act_layer = act_layer
        self.dropout = dropout

    def build(self, input_shape):
        self.in_features = input_shape[-1]
        self.hidden_features = self.hidden_features or self.in_features
        self.out_features = self.out_features or self.in_features
        self.fc1 = tf.keras.layers.Dense(self.hidden_features, name="fc1")
        self.act = tf.keras.layers.Activation(self.act_layer, name="act")
        self.fc2 = tf.keras.layers.Dense(self.out_features, name="fc2")
        self.drop1 = tf.keras.layers.Dropout(self.dropout, name="drop1")
        self.drop2 = tf.keras.layers.Dropout(self.dropout, name="drop2")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_features":self.hidden_features, 
            "out_features":self.out_features, 
            "act_layer":self.act_layer,
            "dropout":self.dropout
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
        self.fc = [
            tf.keras.layers.Dense(int(inp * self.expansion), use_bias=False, name='fc/0'),
            tf.keras.layers.Activation('gelu', name='fc/1'),
            tf.keras.layers.Dense(self.oup, use_bias=False, name='fc/2'),
            tf.keras.layers.Activation('sigmoid', name='fc/3')
            ]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        b, _, _, c = tf.unstack(tf.shape(inputs), num=4)
        x = tf.reshape(self.avg_pool(inputs), (b, c))
        for layer in self.fc:
            x = layer(x)
        x = tf.reshape(x, (b, 1, 1, c))
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
        self.pad1 = tf.keras.layers.ZeroPadding2D(1, name='pad1')
        self.pad2 = tf.keras.layers.ZeroPadding2D(1, name='pad2')
        self.conv = [
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='valid', use_bias=False, name='conv/0'),
            tf.keras.layers.Activation('gelu', name='conv/1'),
            SE(name='conv/2'),
            tf.keras.layers.Conv2D(dim, kernel_size=1, strides=1, padding='valid', use_bias=False, name='conv/3')
        ]
        self.reduction = tf.keras.layers.Conv2D(dim_out, kernel_size=3, strides=2, padding='valid', use_bias=False,
                                                name='reduction')
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm1')  # eps like PyTorch
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm2')
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = self.norm1(inputs)
        xr = self.pad1(x)  # if pad had weights it would've thrown error with .save_weights()
        for layer in self.conv:
            xr = layer(xr)
        x = x + xr
        x = self.pad2(x)
        x = self.reduction(x)
        x = self.norm2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "keep_dim":self.keep_dim,
        })
        return config

@tf.keras.utils.register_keras_serializable(package="gcvit")
class FeatExtract(tf.keras.layers.Layer):
    def __init__(self, keep_dim=False, **kwargs):
        super().__init__(**kwargs)
        self.keep_dim = keep_dim

    def build(self, input_shape):
        dim = input_shape[-1]
        self.pad1 = tf.keras.layers.ZeroPadding2D(1, name='pad1')
        self.pad2 = tf.keras.layers.ZeroPadding2D(1, name='pad2')
        self.conv = [
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='valid', use_bias=False, name='conv/0'),
            tf.keras.layers.Activation('gelu', name='conv/1'),
            SE(name='conv/2'),
            tf.keras.layers.Conv2D(dim, kernel_size=1, strides=1, padding='valid', use_bias=False, name='conv/3')
        ]
        if not self.keep_dim:
            self.pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', name='pool')
        # else:
        #     self.pool = tf.keras.layers.Activation('linear', name='identity')  # hack for PyTorch nn.Identity layer ;)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        xr = self.pad1(x)
        for layer in self.conv:
            xr = layer(xr)
        x = x + xr # if pad had weights it would've thrown error with .save_weights()
        if not self.keep_dim:
            x = self.pad2(x)
            x = self.pool(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "keep_dim":self.keep_dim,
        })
        return config
    
@tf.keras.utils.register_keras_serializable(package="gcvit")    
class Resizing(tf.keras.layers.Layer):
    def __init__(self,
               height,
               width,
               interpolation='bilinear',
               **kwargs):
        self.height = height
        self.width = width
        self.interpolation = interpolation
        super().__init__(**kwargs)

    def call(self, inputs):
        # tf.image.resize will always output float32 and operate more efficiently on
        # float32 unless interpolation is nearest, in which case ouput type matches
        # input type.
        if self.interpolation == 'nearest':
            input_dtype = self.compute_dtype
        else:
            input_dtype = tf.float32
        inputs = tf.cast(inputs, dtype=input_dtype)
        size = [self.height, self.width]
        outputs = tf.image.resize(
            inputs,
            size=size,
            method=self.interpolation)
        return tf.cast(outputs, self.compute_dtype)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = self.height
        input_shape[W_AXIS] = self.width
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'height': self.height,
            'width': self.width,
            'interpolation': self.interpolation,
            })
        return config