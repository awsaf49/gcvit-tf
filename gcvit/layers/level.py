import tensorflow as tf

from .feature import FeatExtract, ReduceSize, Resizing, FitWindow
from .block import GCViTBlock

@tf.keras.utils.register_keras_serializable(package="gcvit")
class GCViTLayer(tf.keras.layers.Layer):
    def __init__(self, depth, num_heads, window_size, keep_dims, downsample=True, mlp_ratio=4., qkv_bias=True, 
                qk_scale=None, drop=0., attn_drop=0., path_drop=0., layer_scale=None, resize_query=False, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.num_heads = num_heads
        self.window_size = window_size
        self.keep_dims = keep_dims
        self.downsample = downsample
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.layer_scale = layer_scale
        self.resize_query = resize_query

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
        self.down = ReduceSize(keep_dim=False, name='downsample')
        self.to_q_global = [
            FeatExtract(keep_dim, name=f'to_q_global/{i}')
            for i, keep_dim in enumerate(self.keep_dims)]
        self.resize = Resizing(self.window_size, self.window_size, interpolation='bicubic')
        self.fit_window = FitWindow(self.window_size)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        H, W = tf.unstack(tf.shape(inputs)[1:3], num=2)
        # pad to fit window_size
        x = self.fit_window(inputs)
        # generate global query
        q_global = x  # (B, H, W, C)
        for layer in self.to_q_global:
            q_global = layer(q_global)  #  official impl issue: https://github.com/NVlabs/GCVit/issues/13
        # resize query to fit key-value, but result in poor score with official weights?
        if self.resize_query:
            q_global = self.resize(q_global)  # to avoid mismatch between feat_map and q_global: https://github.com/NVlabs/GCVit/issues/9
        # feature_map -> windows -> window_attention -> feature_map
        for i, blk in enumerate(self.blocks):
            if i % 2:
                x = blk([x, q_global])
            else:
                x = blk([x])
        x = x[:, :H, :W, :]  # https://github.com/NVlabs/GCVit/issues/9
        # set shape for [B, ?, ?, C]
        x.set_shape(inputs.shape)  # `tf.reshape` creates new tensor with new_shape
        # downsample
        if self.downsample:
          x = self.down(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'depth': self.depth,
            'num_heads': self.num_heads,
            'window_size': self.window_size,
            'keep_dims': self.keep_dims,
            'downsample': self.downsample,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'drop': self.drop,
            'attn_drop': self.attn_drop,
            'path_drop': self.path_drop,
            'layer_scale': self.layer_scale
        })
        return config