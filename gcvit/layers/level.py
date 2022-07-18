import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from .feature import FeatExtract
from .block import GCViTBlock

@register_keras_serializable(package="gcvit")
class GCViTLayer(tf.keras.layers.Layer):
    def __init__(self, depth, num_heads, window_size, keep_dims, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., path_drop=0., layer_scale=None, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.keep_dims = keep_dims
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.layer_scale = layer_scale

    def build(self, input_shape):
        path_drop = [self.path_drop] * self.depth if not isinstance(self.path_drop, list) else self.path_drop
        self.blocks = [
            GCViTBlock(window_size=self.window_size,
                      num_heads=self.num_heads,
                      global_query=bool(i % 2),
                      mlp_ratio=self.mlp_ratio, 
                      qkv_bias=self.qkv_bias, 
                      qk_scale=self.qk_scale, 
                      drop=self.drop,
                      attn_drop=self.attn_drop, 
                      path_drop=path_drop[i],
                      layer_scale=self.layer_scale, 
                      name=f'blocks/{i}')
            for i in range(self.depth)]
        self.to_q_global = [
            FeatExtract(keep_dim, name=f'to_q_global/{i}')
            for i, keep_dim in enumerate(self.keep_dims)]
        self.resize = tf.keras.layers.Resizing(self.window_size, self.window_size, interpolation='bicubic')
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        height, width = tf.shape(inputs)[1:3]
        # pad to multiple of window_size
        h_pad = (self.window_size - height % self.window_size) % self.window_size
        w_pad = (self.window_size - width % self.window_size) % self.window_size
        x = tf.pad(inputs, [[0, 0],
                            [h_pad//2, (h_pad//2 + h_pad%2)],  # padding in both directions unlike tfgcvit
                            [w_pad//2, (w_pad//2 + w_pad%2)],
                            [0, 0]])
        # generate global query
        q_global = x  # (B, H, W, C)
        for layer in self.to_q_global:
            q_global = layer(q_global)  #  official impl issue: https://github.com/NVlabs/GCVit/issues/13
        q_global = self.resize(q_global)  # to avoid mismatch between feat_map and q_global: https://github.com/NVlabs/GCVit/issues/9
        # feature_map -> windows -> window_attention -> feature_map
        for i, blk in enumerate(self.blocks):
            if i % 2:
                x = blk([x, q_global])
            else:
                x = blk([x])
        x = x[:, :height, :width, :]  # https://github.com/NVlabs/GCVit/issues/9
        x.set_shape(inputs.shape)  # `tf.reshape` creates new tensor with new_shape
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'depth': self.depth,
            'num_heads': self.num_heads,
            'window_size': self.window_size,
            'keep_dims': self.keep_dims,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'drop': self.drop,
            'attn_drop': self.attn_drop,
            'path_drop': self.path_drop,
            'layer_scale': self.layer_scale
        })
        return config