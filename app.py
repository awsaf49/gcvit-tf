import gradio as gr
import gcvit
from gcvit.utils import get_gradcam_model, get_gradcam_prediction


def predict_fn(image, model_name):
    """A predict function that will be invoked by gradio."""
    model = getattr(gcvit, model_name)(pretrain=True)
    gradcam_model = get_gradcam_model(model)
    preds, overlay = get_gradcam_prediction(
        image, gradcam_model, cmap="jet", alpha=0.4, pred_index=None
    )
    preds = {x[1]: float(x[2]) for x in preds}
    return [preds, overlay]


demo = gr.Interface(
    fn=predict_fn,
    inputs=[
        gr.inputs.Image(label="Input Image"),
        gr.Radio(
            ["GCViTTiny", "GCViTSmall", "GCViTBase"],
            value="GCViTTiny",
            label="Model Size",
        ),
    ],
    outputs=[
        gr.outputs.Label(label="Prediction"),
        gr.inputs.Image(label="GradCAM"),
    ],
    title="Global Context Vision Transformer (GCViT) Demo",
    description="Image Classification with ImageNet Pretrain Models.",
    examples=[
        ["example/hot_air_ballon.jpg", "GCViTTiny"],
        ["example/chelsea.png", "GCViTTiny"],
        ["example/german_shepherd.jpg", "GCViTTiny"],
        ["example/panda.jpg", "GCViTTiny"],
        ["example/jellyfish.jpg", "GCViTTiny"],
        ["example/penguin.JPG", "GCViTTiny"],
        ["example/bus.jpg", "GCViTTiny"],
        ["example/cat_dog.JPG", "GCViTTiny"],
    ],
)
demo.launch()
