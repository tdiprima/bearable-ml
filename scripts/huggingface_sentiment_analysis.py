"""
This script uses a pre-trained Hugging Face model to analyze the sentiment of text.

What it does:
- Loads an existing sentiment analysis model
- Runs text through it
- Outputs whether the text sounds positive, negative, etc.

No training happens here — this is just using a ready-made model.
"""

from transformers import pipeline

from my_timer import timer

with timer("Hugging Face sentiment analysis"):
    # Load the pipeline
    classifier = pipeline("sentiment-analysis")

    # Perform prediction
    result = classifier("Everything went smoothly and I'm pleased with the result.")

    # Print result
    print(result)

    result = classifier("The setup was confusing and the results were unreliable.")
    print(result)
