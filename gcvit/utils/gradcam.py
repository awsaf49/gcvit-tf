import tensorflow as tf
import matplotlib.cm as cm
import numpy as np

def process_image(img, size=(224, 224)):
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(img, mode='torch')
    img_array = tf.image.resize(img_array, size,)[None,]
    return img_array

def get_gradcam_model(model):
    inp = tf.keras.Input(shape=(224, 224, 3))
    feats = model.forward_features(inp)
    preds = model.forward_head(feats)
    return tf.keras.models.Model(inp, [preds, feats])

def get_gradcam_prediction(img, grad_model, pred_index=None, cmap='jet', alpha=0.4):
    # process image for inference
    img_array = process_image(img)
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
    heatmap = tf.keras.utils.array_to_img(heatmap)
    heatmap = heatmap.resize((img.shape[1], img.shape[0]))
    heatmap = tf.keras.utils.img_to_array(heatmap)
    overlay = img + heatmap * alpha
    overlay = tf.keras.utils.array_to_img(overlay)
    # decode prediction
    preds_decode = tf.keras.applications.imagenet_utils.decode_predictions(preds.numpy())[0]
    return preds_decode, overlay