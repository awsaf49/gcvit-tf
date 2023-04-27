import tensorflow as tf
from tensorflow.keras import mixed_precision
import os
from gcvit import GCViTTiny
from skimage.data import chelsea
import numpy as np


print("tf version:", tf.version.VERSION)


def load_image():
    img = tf.keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch')  # Chelsea the cat
    img = tf.image.resize(img, (224, 224))[None,] # resize & create batch
    return img


def test_mixed_precision():
    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # enable mixed precision
    mixed_precision.set_global_policy("mixed_float16")

    # define model architecture
    model = GCViTTiny(pretrain=True)
    model.reset_classifier(num_classes=1, head_act="sigmoid")
    print(model.summary())

    # make toy dataset
    data = np.concatenate([load_image() for i in range(8)], axis=0)
    gt = np.expand_dims(np.random.randint(2, size=8), axis=-1)

    print(data.shape)
    print(gt.shape)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["acc"],
    )

    # test that training works
    model.fit(
        data,
        gt,
        epochs=1,
        batch_size=1,
        verbose=1,
    )

    # test that inference works
    pred = model(data[:1]).numpy()
    print(pred)


if __name__ == "__main__":
    test_mixed_precision()
