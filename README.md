# bearable-ml 🐻

A collection of small, approachable machine learning experiments

This repository is a collection of **small, focused ML scripts** meant to be  
easy to read, easy to run, and easy to understand — without requiring  
research-paper energy.

The goal is not to build production systems, but to explore **how different ML  
tools work in practice**, one script at a time.

Inspired by [an article](https://medium.com/python-in-plain-english/the-7-python-libraries-that-turn-your-model-training-into-a-single-line-of-code-e2c6bab56a4c) on automating model training.

## What's in here

Each script demonstrates a specific tool or concept, including:

- Automated model training and comparison (AutoGluon, PyCaret)
- Hyperparameter optimization (Optuna)
- Experiment tracking (MLflow)
- Image classification (FastAI / ResNet)
- Text sentiment analysis (Hugging Face)
- Time series forecasting (NeuralProphet)

Most scripts are intentionally small and self-contained.

## What this repo is (and isn't)

**This repo is:**

- A learning lab
- A reference for "how do I even start with this tool?"
- Focused on clarity over cleverness

**This repo is not:**

- A production-ready ML system
- An opinionated framework
- A benchmark or leaderboard flex

## Requirements

- Python 3.11+
- A virtual environment is strongly recommended

Individual scripts may require different libraries. See comments at the top of
each file for details.

Some scripts (especially image models) may run **very slowly on CPU-only  
machines**.

## Running the examples

Most scripts can be run directly:

```bash
python path/to/script.py
```

Some scripts will download small sample datasets automatically.  
Others expect local data — see the script header comments for guidance.

## Repo structure

```text
.
├── data/          # Small sample datasets or images
├── scripts/       # Individual ML examples
├── README.md
└── requirements.txt
```

## Why "bearable"?

Machine learning doesn't need to feel scary, academic, or inaccessible.  
These examples are written for **future me**, tired me, and curious me.

If it's readable, runnable, and understandable — it belongs here.

## License

[MIT](LICENSE)

<br>
