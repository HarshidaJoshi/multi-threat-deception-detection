"""Real Grad-CAM for the trained image classifiers (EfficientNetB0 / ResNet50 backbones).

Computes d(fake probability) / d(last convolutional feature map), weights the feature channels by
the spatially averaged gradients, and overlays the resulting class-activation map on the image.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

INPUT_SIZE = (224, 224)


def _split_model(model):
    """Return (pre, base, head_layers): preprocessing sub-model, nested backbone, layers after it."""
    import tensorflow as tf

    layers = list(model.layers)
    base_idx = next(i for i, l in enumerate(layers) if isinstance(l, tf.keras.Model))
    base = layers[base_idx]
    # Whatever the saved graph does between the raw input and the backbone (e.g. ResNet50's
    # channel flip / mean subtraction) must be replayed, or the explanation would not match the model.
    feeding = base._inbound_nodes[0].input_tensors[0]
    pre = tf.keras.Model(model.input, feeding)
    return pre, base, layers[base_idx + 1:]


def _last_conv_layer(base):
    for layer in reversed(base.layers):
        shape = getattr(layer, "output", None)
        shape = getattr(shape, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer
    raise ValueError("No 4-D convolutional feature map found in backbone")


def _forward_factory(model):
    import tensorflow as tf

    pre, base, head = _split_model(model)
    conv_layer = _last_conv_layer(base)
    feat_model = tf.keras.Model(base.input, [conv_layer.output, base.output])

    def forward(x):
        conv_out, pooled = feat_model(pre(x, training=False), training=False)
        h = pooled
        for layer in head:
            h = layer(h, training=False)
        return conv_out, h

    return forward, conv_layer.name


def gradcam_map(model, batch: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Return (heatmap in [0,1] at feature-map resolution, fake probability, conv layer name)."""
    import tensorflow as tf

    forward, layer_name = _forward_factory(model)
    x = tf.convert_to_tensor(batch)
    with tf.GradientTape() as tape:
        conv_out, prob = forward(x)
        tape.watch(conv_out)
        score = prob[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, float(prob[0, 0]), layer_name


def _jet(v: np.ndarray) -> np.ndarray:
    """Small jet-like colormap (no matplotlib dependency)."""
    v = np.clip(v, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def gradcam_overlay_base64(model, image_path: str, alpha: float = 0.45) -> dict:
    """Grad-CAM overlay for ``image_path`` as a base64 PNG plus metadata."""
    import tensorflow as tf

    img = Image.open(image_path).convert("RGB")
    # Same resize as training/inference so the explanation is of the prediction the app reports.
    batch = np.expand_dims(tf.image.resize(tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3,
                                                              expand_animations=False), INPUT_SIZE).numpy().astype(np.float32), 0)
    cam, prob, layer_name = gradcam_map(model, batch)

    show = img.copy()
    show.thumbnail((384, 384), Image.Resampling.LANCZOS)
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(show.size, Image.Resampling.BICUBIC)
    heat = _jet(np.asarray(cam_img, dtype=np.float32) / 255.0)
    base = np.asarray(show, dtype=np.float32) / 255.0
    overlay = np.clip((1 - alpha) * base + alpha * heat, 0, 1)
    buf = io.BytesIO()
    Image.fromarray((overlay * 255).astype(np.uint8)).save(buf, format="PNG")
    return {
        "png_base64": base64.b64encode(buf.getvalue()).decode(),
        "fake_probability": prob,
        "layer": layer_name,
        "peak_fraction": float((cam > 0.6).mean()),
    }
