import numpy as np
import cv2
import tensorflow as tf


def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    base_model = model.layers[0]

    grad_model = tf.keras.models.Model(
        [base_model.inputs],
        [base_model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img_path, heatmap, output_path, alpha=0.45):
    img = cv2.imread(original_img_path)
    img = cv2.resize(img, (300, 300))

    heatmap_resized = cv2.resize(heatmap, (300, 300))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    overlayed = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)
    cv2.imwrite(output_path, overlayed)