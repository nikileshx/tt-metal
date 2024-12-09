# Alexnet ttnn Implementation


# Platforms:
    GS E150, WH N150, WH N300


## Introduction
AlexNet is a deep learning model designed for image classification tasks. It uses convolutional layers to extract features from images and classify them accurately. Known for introducing ReLU activations, dropout, and GPU acceleration, AlexNet played a key role in advancing deep learning and remains a foundational model in computer vision.

## Batch Size
Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the batch_size to 4.

# Details
The entry point to alexnet model is ttnn_alexnet in `models/experimental/functional_alexnet/tt/ttnn_alexnet.py`.

## How to Run

To run on MNIST dataset:
```
pytest --disable-warnings models/experimental/functional_alexnet/demo/demo.py::test_alexnet_on_mnist
```
To run on Image folder:
```
pytest --disable-warnings models/experimental/functional_alexnet/demo/demo.py::test_alexnet_on_imageFolder
```

## Inputs

The demo receives input from MNIST datset and downloaded images.
