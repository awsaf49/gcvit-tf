import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable
from ..layers import PatchEmbed, GCViTLayer, Identity


@register_keras_serializable(package='gcvit')
class GCViT(tf.keras.Model):
    def __init__(self, window_size, dim, depths, num_heads,
        drop_rate=0., mlp_ratio=3., qkv_bias=True, qk_scale=None, attn_drop=0., path_drop=0.1, layer_scale=None,
        pooling=None, classes=1000, classifier_activation='softmax', **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.dim = dim
        self.depths = depths
        self.num_heads = num_heads
        self.drop_rate = drop_rate
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.layer_scale = layer_scale
        self.pooling = pooling
        self.classes = classes
        self.classifier_activation = classifier_activation

        self.patch_embed = PatchEmbed(dim=dim, name='patch_embed')
        self.pos_drop = tf.keras.layers.Dropout(drop_rate, name='pos_drop')
        path_drops = np.linspace(0., path_drop, sum(depths))
        keep_dims = [(False, False, False),(False, False),(True,),(True,),]
        self.levels = []
        for i in range(len(depths)):
            path_drop = path_drops[sum(depths[:i]):sum(depths[:i + 1])].tolist()
            level = GCViTLayer(depth=depths[i], num_heads=num_heads[i], window_size=window_size[i], keep_dims=keep_dims[i],
                    downsample=(i < len(depths) - 1), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    drop=drop_rate, attn_drop=attn_drop, path_drop=path_drop, layer_scale=layer_scale, name=f'levels/{i}')
            self.levels.append(level)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm')
        if pooling == 'avg':
            self.pool = tf.keras.layers.GlobalAveragePooling2D(name='pool')
        elif pooling == 'max':
            self.pool = tf.keras.layers.GlobalMaxPooling2D(name='pool')
        elif pooling is None:
            self.pool = Identity(name='pool')
        else:
            raise ValueError(f'Expecting pooling to be one of None/avg/max. Found: {pooling}')
        self.head = tf.keras.layers.Dense(classes, name='head')
        self.head_act = tf.keras.layers.Activation(classifier_activation, name='head_act')

    def feature(self, inputs, **kwargs):
        # Define model pipeline
        x = self.patch_embed(inputs)
        x = self.pos_drop(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.pool(x)
        return x

    def call(self, inputs, **kwargs):
        x = self.feature(inputs)
        x = self.head(x)
        x = self.head_act(x)
        return x
    
    def get_config(self):
        config = {
            'window_size': self.window_size,
            'dim': self.dim,
            'depths': self.depths,
            'num_heads': self.num_heads,
            'drop_rate': self.drop_rate,
            'mlp_ratio': self.mlp_ratio,
            'qkv_bias': self.qkv_bias,
            'qk_scale': self.qk_scale,
            'attn_drop': self.attn_drop,
            'path_drop': self.path_drop,
            'layer_scale': self.layer_scale,
            'pooling': self.pooling,
            'classes': self.classes,
            'classifier_activation': self.classifier_activation
        }
        return config
            
