"""
This script fine-tunes a pre-trained ResNet34 model using FastAI
to classify images (for example, different kinds of pets).

What it does:
- Downloads a sample pet image dataset
- Trains an image classifier on those images
- Lets you run predictions on your own images afterward

Note:
This is compute-heavy and will run much faster on a GPU.
It may be slow or painful on a CPU-only system.
"""

from fastai.vision.all import *

from my_timer import timer

with timer("FastAI image classification"):
    # Download and prepare data
    path = untar_data(URLs.PETS) / "images"

    # Define label function
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        label_func=lambda x: "cat" if x[0].isupper() else "dog",  # because that's how the data comes in
        item_tfms=Resize(224),
    )

    # Train and fine-tune the model
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(1)

    # Example prediction
    img = PILImage.create("../data/cat.jpg")

    pred, pred_idx, probs = learn.predict(img)
    print(f"Prediction: {pred}; Probability: {probs[pred_idx]:.4f}")
