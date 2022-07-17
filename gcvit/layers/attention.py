import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package='TFGCVit')
class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_dropout=0., proj_dropout=0.,
                 **kwargs):
        super().__init__(**kwargs)
        window_size = (window_size,window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

    def build(self, input_shape):
        dim = input_shape[-1]
        head_dim = dim // self.num_heads
        self.scale = self.qk_scale or head_dim ** -0.5
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=self.qkv_bias, name='qkv')
        self.relative_position_bias_table = self.add_weight(
            'relative_position_bias_table',
            shape=[(2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            dtype=self.dtype)
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'), axis=0)
        coords_flatten = tf.reshape(coords, [2, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        relative_coords_xx = (relative_coords[:, :, 0] + self.window_size[0] - 1)
        relative_coords_yy = (relative_coords[:, :, 1] + self.window_size[1] - 1) 
        relative_coords_xx = relative_coords_xx * (2 * self.window_size[1] - 1)
        relative_position_index = (relative_coords_xx + relative_coords_yy)
        self.relative_position_index = tf.Variable(relative_position_index, 
                                                  trainable=False,
                                                  name=f'relative_position_index')
        self.attn_drop = tf.keras.layers.Dropout(self.attn_dropout, name='attn_drop')
        self.proj = tf.keras.layers.Dense(dim, name='proj')
        self.proj_drop = tf.keras.layers.Dropout(self.proj_dropout, name='proj_drop')
        self.softmax = tf.keras.layers.Activation('softmax', name='softmax')
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        B_, N, C = tf.shape(inputs)
        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, axis=0)
        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias,
                                            shape=[self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        attn = attn + relative_position_bias[tf.newaxis,]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3]) # B, num_token, num_head, channels_per_head
        x = tf.reshape(x, shape=[B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'window_size': self.window_size,
            'num_heads': self.num_heads,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'attn_dropout': self.attn_droput,
            'proj_dropout': self.proj_droput
        })

        return config