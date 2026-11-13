# CNN Shape Classifier

A Convolutional Neural Network that classifies 28x28 grayscale images as a **circle**, **square**, or **triangle**.

## What it shows

- Convolution, ReLU, and max-pooling layers that extract image features
- A fully connected classifier on top of those features
- Training with cross-entropy loss and the Adam optimizer

## Setup

```bash
pip install torch
```

## Run

```bash
python cnn.py
```

The script trains for 8 epochs, prints test accuracy after each epoch, and saves the model to `cnn_model.pt`. No dataset download is needed; the shapes are drawn in code with random size, position, and noise.

## Files

- `cnn.py`: shape generation, the CNN model, and the training loop
- `requirements.txt`: Python dependencies
