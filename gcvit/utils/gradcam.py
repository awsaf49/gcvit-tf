import tensorflow as tf
import matplotlib.cm as cm
import numpy as np
try:
    from tensorflow.keras.utils import array_to_img, img_to_array
except:
    from tensorflow.keras.preprocessing.image import array_to_img, img_to_array

def process_image(img, size=(224, 224)):
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(img, mode='torch')
    img_array = tf.image.resize(img_array, size,)[None,]
    return img_array

def get_gradcam_model(model):
    inp = tf.keras.Input(shape=(224, 224, 3))
    feats = model.forward_features(inp)
    preds = model.forward_head(feats)
    return tf.keras.models.Model(inp, [preds, feats])

def get_gradcam_prediction(img, grad_model, process=True, decode=True, pred_index=None, cmap='jet', alpha=0.4):
    """Grad-CAM for a single image

    Args:
        img (np.ndarray): process or raw image without batch_shape e.g. (224, 224, 3)
        grad_model (tf.keras.Model): model with feature map and prediction
        process (bool, optional): imagenet pre-processing. Defaults to True.
        pred_index (int, optional): for particular calss. Defaults to None.
        cmap (str, optional): colormap. Defaults to 'jet'.
        alpha (float, optional): opacity. Defaults to 0.4.

    Returns:
        preds_decode: top5 predictions
        heatmap: gradcam heatmap
    """
    # process image for inference
    if process:
        img_array = process_image(img)
    else:
        img_array = tf.convert_to_tensor(img)[None,]
        if img.min()!=img.max():
            img = (img - img.min())/(img.max() - img.min())
            img = np.uint8(img*255.0)
    # get prediction
    with tf.GradientTape(persistent=True) as tape:
        preds, feats = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    # compute heatmap
    grads = tape.gradient(class_channel, feats)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    feats = feats[0]
    heatmap = feats @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = np.uint8(255 * heatmap)
    # colorize heatmap
    cmap = cm.get_cmap(cmap)
    colors = cmap(np.arange(256))[:, :3]
    heatmap = colors[heatmap]
    heatmap = array_to_img(heatmap)
    heatmap = heatmap.resize((img.shape[1], img.shape[0]))
    heatmap = img_to_array(heatmap)
    overlay = img + heatmap * alpha
    overlay = array_to_img(overlay)
    # decode prediction
    preds_decode = tf.keras.applications.imagenet_utils.decode_predictions(preds.numpy())[0] if decode else preds
    return preds_decode, overlay