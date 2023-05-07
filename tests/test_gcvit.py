import numpy as np
import tensorflow as tf
from skimage.data import chelsea

from gcvit import GCViTTiny


def test_inference():
    # load test image, resize, and create batch
    img = tf.keras.applications.imagenet_utils.preprocess_input(
        chelsea(), mode="torch"
    )
    img = tf.image.resize(img, (224, 224))[
        None,
    ]

    model = GCViTTiny(pretrain=True)

    pred = model(img).numpy()
    result = tf.keras.applications.imagenet_utils.decode_predictions(pred)[0]
    print(result)

    # should be five predictions for this pretrained model and image
    assert len(result) == 5

    # check if values match with expected results
    expected_result = [
        ("n02124075", "Egyptian_cat", 0.7402193),
        ("n02123045", "tabby", 0.07757583),
        ("n02123159", "tiger_cat", 0.07006007),
        ("n02127052", "lynx", 0.0043750308),
        ("n04040759", "radiator", 0.00091264246),
    ]
    for res, exp in zip(result, expected_result):
        assert res[1] == exp[1]
        assert np.around(res[2], decimals=4) == np.float32(
            np.around(exp[2], decimals=4)
        )


def test_feature_extraction():
    # load test image, resize, and create batch
    img = tf.keras.applications.imagenet_utils.preprocess_input(
        chelsea(), mode="torch"
    )
    img = tf.image.resize(img, (224, 224))[
        None,
    ]

    model = GCViTTiny(pretrain=True)
    model.reset_classifier(num_classes=0, head_act=None)
    feature = model(img)
    shape_ = feature.shape
    print(shape_)

    assert shape_ == (1, 512)
