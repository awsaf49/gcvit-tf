import math
from typing import Tuple

import tensorflow as tf


class AdaptiveAveragePooling2D(tf.keras.layers.Layer):
    def __init__(
        self,
        output_size: Tuple[int, int],
        input_ordering: str = "NHWC",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.input_ordering = input_ordering
        if input_ordering not in ("NCHW", "NHWC"):
            raise ValueError(
                "Unrecognized input_ordering, should be 'NCHW' or 'NHWC'!"
            )
        self.h_axis = input_ordering.index("H")
        self.w_axis = input_ordering.index("W")

    def pseudo_1d_pool(self, inputs: tf.Tensor, h_pooling: bool):
        # Figure out which axis we're pooling on
        if h_pooling:
            axis = self.h_axis
            output_dim = self.output_size[0]
        else:
            axis = self.w_axis
            output_dim = self.output_size[1]
        input_dim = inputs.shape[axis]

        # Figure out the potential pooling windows
        # This is the key idea - the torch op will always use only two
        # consecutive pooling window sizes, like 3 and 4. Therefore,
        # if we pool with both possible sizes, we simply need to gather
        # the 'correct' pool at each position to reimplement the torch op.
        small_window = math.ceil(input_dim / output_dim)
        big_window = small_window + 1
        if h_pooling:
            output_dim = self.output_size[0]
            small_window_shape = (small_window, 1)
            big_window_shape = (big_window, 1)
        else:
            output_dim = self.output_size[1]
            small_window_shape = (1, small_window)
            big_window_shape = (1, big_window)

        # For integer resizes, we can take a very quick shortcut
        if input_dim % output_dim == 0:
            return tf.nn.avg_pool2d(
                inputs,
                ksize=small_window_shape,
                strides=small_window_shape,
                padding="VALID",
                data_format=self.input_ordering,
            )

        # For non-integer resizes, we pool with both possible window sizes
        # and concatenate them
        small_pool = tf.nn.avg_pool2d(
            inputs,
            ksize=small_window_shape,
            strides=1,
            padding="VALID",
            data_format=self.input_ordering,
        )
        big_pool = tf.nn.avg_pool2d(
            inputs,
            ksize=big_window_shape,
            strides=1,
            padding="VALID",
            data_format=self.input_ordering,
        )
        both_pool = tf.concat([small_pool, big_pool], axis=axis)

        # We compute vectors of the start and end positions
        # for each pooling window
        # Each (start, end) pair here corresponds to a single output position
        window_starts = tf.math.floor(
            (tf.range(output_dim, dtype=tf.float32) * input_dim) / output_dim
        )
        window_starts = tf.cast(window_starts, tf.int64)
        window_ends = tf.math.ceil(
            (tf.range(1, output_dim + 1, dtype=tf.float32) * input_dim)
            / output_dim
        )
        window_ends = tf.cast(window_ends, tf.int64)

        # pool_selector is a boolean array of shape (output_dim,)
        # where 1 indicates that output position
        # has a big receptive field and 0 indicates that that output
        # position has a small receptive field
        pool_selector = tf.cast(
            window_ends - window_starts - small_window, tf.bool
        )

        # Since we concatenated the small and big pools, we need to do a bit of
        # pointer arithmetic to get the indices of the big pools
        small_indices = window_starts
        big_indices = window_starts + small_pool.shape[axis]

        # Finally, we use the pool_selector to generate a list of indices,
        # one per output position
        gather_indices = tf.where(pool_selector, big_indices, small_indices)

        # Gathering from those indices yields the final, correct pooling
        return tf.gather(both_pool, gather_indices, axis=axis)

    def call(self, inputs: tf.Tensor):
        if self.input_ordering == "NHWC":
            input_shape = inputs.shape[1:3]
        else:
            input_shape = inputs.shape[2:]

        if (
            input_shape[0] % self.output_size[0] == 0
            and input_shape[1] % self.output_size[1] == 0
        ):
            # If we're resizing by an integer factor on both dimensions,
            # we can take a very quick shortcut.
            h_resize = int(input_shape[0] // self.output_size[0])
            w_resize = int(input_shape[1] // self.output_size[1])
            return tf.nn.avg_pool2d(
                inputs,
                ksize=(h_resize, w_resize),
                strides=(h_resize, w_resize),
                padding="VALID",
                data_format=self.input_ordering,
            )
        else:
            # If we can't take the shortcut, we do a 1D pool on each axis
            h_pooled = self.pseudo_1d_pool(inputs, h_pooling=True)
            return self.pseudo_1d_pool(h_pooled, h_pooling=False)
