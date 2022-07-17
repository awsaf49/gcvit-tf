import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from .attention import WindowAttentionGlobal, WindowAttention
from .drop import DropPath, Identity
from .window import window_partition, window_reverse
from .feature import Mlp


@register_keras_serializable(package="gcvit")
class GCViTBlock(tf.keras.layers.Layer):
    def __init__(self, window_size, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., path_drop=0., act_layer='gelu', attention='global',
                 layer_scale=None, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.act_layer = act_layer
        self.attention = attention
        self.layer_scale = layer_scale

    def build(self, input_shape):
        B, H, W, C = input_shape[0]
        self.norm1 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm1')
        self.attn_layer = WindowAttentionGlobal if self.attention == 'global' else WindowAttention
        self.attn = self.attn_layer(window_size=self.window_size, 
                                   num_heads=self.num_heads,
                                   qkv_bias=self.qkv_bias, 
                                   qk_scale=self.qk_scale, 
                                   attn_dropout=self.attn_drop, 
                                   proj_dropout=self.drop,
                                   name='attn')
        self.drop_path = DropPath(self.path_drop)
        self.norm2 = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm2')
        self.mlp = Mlp(hidden_features=int(C * self.mlp_ratio), dropout=self.drop, act_layer=self.act_layer, name='mlp')
        if self.layer_scale is not None:
            self.gamma1 = self.add_weight(
                'gamma1',
                shape=[C],
                initializer=tf.keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype)
            self.gamma2 = self.add_weight(
                'gamma2',
                shape=[C],
                initializer=tf.keras.initializers.Constant(self.layer_scale),
                trainable=True,
                dtype=self.dtype)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        self.num_windows = int(H // self.window_size) * int(W // self.window_size)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        inputs,  query = inputs
        B, H, W, C = tf.shape(inputs)
        x = self.norm1(inputs)
        # window partition
        x = window_partition(x, self.window_size)
        # flatten patch
        x = tf.reshape(x, shape=[-1, self.window_size * self.window_size, C])
        # attention
        x = self.attn([x, query])
        # reverse window partition
        x = window_reverse(x, self.window_size, H, W, C)
        # FFN
        x = inputs + self.drop_path(x * self.gamma1)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'window_size': self.window_size,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'drop': self.drop,
            'attn_drop': self.attn_drop,
            'path_drop': self.path_drop,
            'act_layer': self.act_layer,
            'attention': self.attention,
            'layer_scale': self.layer_scale,
            'num_windows': self.num_windows,
        })
        return config