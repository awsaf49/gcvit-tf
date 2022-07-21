import numpy as np
import tensorflow as tf

from ..layers import PatchEmbed, GCViTLayer, Identity


BASE_URL = 'https://github.com/awsaf49/gcvit-tf/releases/download'
TAG = 'v1.0.4'
NAME2CONFIG = {
    'gcvit_tiny': {'window_size': (7, 7, 14, 7),
                    'dim': 64,
                    'depths': (3, 4, 19, 5),
                    'num_heads': (2, 4, 8, 16), 
                    'path_drop': 0.2,},
    'gcvit_small': {'window_size': (7, 7, 14, 7), 
                     'dim': 96, 
                     'depths': (3, 4, 19, 5),
                     'num_heads': (3, 6, 12, 24),
                     'mlp_ratio': 2.,
                     'path_drop': 0.3,
                     'layer_scale': 1e-5,},
    'gcvit_base': {'window_size': (7, 7, 14, 7),
                    'dim':128, 
                    'depths': (3, 4, 19, 5),
                    'num_heads': (4, 8, 16, 32),
                    'mlp_ratio': 2.,
                    'path_drop': 0.5,
                    'layer_scale': 1e-5,},
    }

@tf.keras.utils.register_keras_serializable(package='gcvit')
class GCViT(tf.keras.Model):
    def __init__(self, window_size, dim, depths, num_heads,
        drop_rate=0., mlp_ratio=3., qkv_bias=True, qk_scale=None, attn_drop=0., path_drop=0.1, layer_scale=None, resize_query=False,
        global_pool='avg', num_classes=1000, head_act='softmax', **kwargs):
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
        self.resize_query = resize_query
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.head_act = head_act

        self.patch_embed = PatchEmbed(dim=dim, name='patch_embed')
        self.pos_drop = tf.keras.layers.Dropout(drop_rate, name='pos_drop')
        path_drops = np.linspace(0., path_drop, sum(depths))
        keep_dims = [(False, False, False),(False, False),(True,),(True,),]
        self.levels = []
        for i in range(len(depths)):
            path_drop = path_drops[sum(depths[:i]):sum(depths[:i + 1])].tolist()
            level = GCViTLayer(depth=depths[i], num_heads=num_heads[i], window_size=window_size[i], keep_dims=keep_dims[i],
                    downsample=(i < len(depths) - 1), mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    drop=drop_rate, attn_drop=attn_drop, path_drop=path_drop, layer_scale=layer_scale, resize_query=resize_query,
                    name=f'levels/{i}')
            self.levels.append(level)
        self.norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-05, name='norm')
        if global_pool == 'avg':
            self.pool = tf.keras.layers.GlobalAveragePooling2D(name='pool')
        elif global_pool == 'max':
            self.pool = tf.keras.layers.GlobalMaxPooling2D(name='pool')
        elif global_pool is None:
            self.pool = Identity(name='pool')
        else:
            raise ValueError(f'Expecting pooling to be one of None/avg/max. Found: {global_pool}')
        self.head = [tf.keras.layers.Dense(num_classes, name='head/fc'),
                     tf.keras.layers.Activation(head_act, name='head/act')]

    def reset_classifier(self, num_classes, head_act, global_pool=None, in_channels=3):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head[0] = tf.keras.layers.Dense(num_classes, name='head/fc') if num_classes else Identity(name='head/fc')
        self.head[1] = tf.keras.layers.Activation(head_act, name='head/act') if head_act else Identity(name='head/act')
        super().build((1, 224, 224, in_channels)) # for head we only need info from the input channel
        
    def forward_features(self, inputs):
        x = self.patch_embed(inputs)
        x = self.pos_drop(x)
        x = tf.cast(x, dtype=tf.float32)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        return x

    def forward_head(self, inputs, pre_logits=False):
        x = inputs
        if self.global_pool in ['avg', 'max']:
            x = self.pool(x)
        if not pre_logits:
            for layer in self.head:
                x = layer(x)
        return x 

    def call(self, inputs, **kwargs):
        x = self.forward_features(inputs)
        x = self.forward_head(x)
        return x

    def build_graph(self, input_shape=(224, 224, 3)):
        """https://www.kaggle.com/code/ipythonx/tf-hybrid-efficientnet-swin-transformer-gradcam"""
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x), name=self.name)

# load standard models
def GCViTTiny(input_shape=(224, 224, 3), pretrain=False, resize_query=False, **kwargs):
    name = 'gcvit_tiny'
    config = NAME2CONFIG[name]
    ckpt_link = '{}/{}/{}_weights.h5'.format(BASE_URL, TAG, name)
    model = GCViT(name=name, resize_query=resize_query, **config,  **kwargs)
    model(tf.random.uniform(shape=input_shape)[tf.newaxis,])
    if pretrain:
        ckpt_path = tf.keras.utils.get_file('{}_weights.h5'.format(name), ckpt_link)
        model.load_weights(ckpt_path)
    return model

def GCViTSmall(input_shape=(224, 224, 3), pretrain=False, resize_query=False, **kwargs):
    name = 'gcvit_small'
    config = NAME2CONFIG[name]
    ckpt_link = '{}/{}/{}_weights.h5'.format(BASE_URL, TAG, name)
    model = GCViT(name=name, resize_query=resize_query, **config, **kwargs)
    model(tf.random.uniform(shape=input_shape)[tf.newaxis,])
    if pretrain:
        ckpt_path = tf.keras.utils.get_file('{}_weights.h5'.format(name), ckpt_link)
        model.load_weights(ckpt_path)
    return model

def GCViTBase(input_shape=(224, 224, 3), pretrain=False, resize_query=False, **kwargs):
    name = 'gcvit_base'
    config = NAME2CONFIG[name]
    ckpt_link = '{}/{}/{}_weights.h5'.format(BASE_URL, TAG, name)
    model = GCViT(name=name, resize_query=resize_query, **config, **kwargs)
    model(tf.random.uniform(shape=input_shape)[tf.newaxis,])
    if pretrain:
        ckpt_path = tf.keras.utils.get_file('{}_weights.h5'.format(name), ckpt_link)
        model.load_weights(ckpt_path)
    return model
