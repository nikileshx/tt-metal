# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import random
import pytest
import torch, ttnn
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_alexnet.tt.ttnn_alexnet import ttnn_alexnet
from models.utility_functions import disable_persistent_kernel_cache, disable_compilation_reports
from models.experimental.functional_alexnet.tt.ttnn_alexnet_utils import custom_preprocessor


def get_dataset(batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    folder_path = "models/experimental/functional_alexnet/demo/images"

    batch_size = min(batch_size, 4)

    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))
    ][:batch_size]

    # Shuffle the file list
    random.shuffle(image_files)

    tensors = []
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")  # Convert to RGB
        tensor = transform(image)
        tensors.append(tensor)

    # Stack tensors into a single batch
    batch = torch.stack(tensors)  # Shape: (num_images, channels, height, width)

    return batch


def run_alexnet_on_imageFolder(device, batch_size):
    disable_persistent_kernel_cache()

    torch_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        convert_to_ttnn=lambda *_: True,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )

    test_input = get_dataset(batch_size=batch_size)
    ttnn_input = test_input.permute((0, 2, 3, 1))

    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with torch.inference_mode():
        ttnn_output_tensor = ttnn_alexnet(device, ttnn_input, parameters)
        ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)
        ttnn_predicted_probabilities = torch.nn.functional.softmax(ttnn_output_tensor, dim=1)
        _, ttnn_predicted_labels = torch.max(ttnn_predicted_probabilities, 1)

    with torch.inference_mode():
        torch_output_tensor = torch_model(test_input)
        torch_predicted_probabilities = torch.nn.functional.softmax(torch_output_tensor, dim=1)
        _, torch_predicted_labels = torch.max(torch_predicted_probabilities, 1)

    batch_size = len(test_input)
    correct = 0
    for i in range(batch_size):
        if torch_predicted_labels[i] == ttnn_predicted_labels[i]:
            correct += 1

    accuracy = correct / (batch_size)

    logger.info(f"Accuracy for {batch_size} Samples : {accuracy}")
    assert accuracy >= 0.998, f"Expected accuracy : {0.998} Actual accuracy: {accuracy}"

    logger.info(f"torch_predicted {torch_predicted_labels}")
    logger.info(f"ttnn_predicted {ttnn_predicted_labels}")


def run_alexnet_on_mnist(device, batch_size, iterations):
    disable_persistent_kernel_cache()

    torch_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    torch_model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        convert_to_ttnn=lambda *_: True,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )

    correct = 0
    for iters in range(iterations):
        dataloader = DataLoader(test_dataset, batch_size=batch_size)
        x, labels = next(iter(dataloader))
        dataset_ttnn_correct = 0

        # ttnn predictions
        ttnn_input = x.permute((0, 2, 3, 1))
        ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        ttnn_output_tensor = ttnn_alexnet(device, ttnn_input, parameters)
        ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)
        ttnn_predicted_probabilities = torch.nn.functional.softmax(ttnn_output_tensor, dim=1)
        _, ttnn_predicted_labels = torch.max(ttnn_predicted_probabilities, 1)

        # torch predictions
        torch_output_tensor = torch_model(x)
        torch_predicted_probabilities = torch.nn.functional.softmax(torch_output_tensor, dim=1)
        _, torch_predicted_labels = torch.max(torch_predicted_probabilities, 1)

        for i in range(batch_size):
            logger.info(f"Iter: {iters} Sample {i}:")
            logger.info(f"torch predicted Label: {torch_predicted_labels[i]}")
            logger.info(f"ttnn predicted Label: {ttnn_predicted_labels[i]}")
            if torch_predicted_labels[i] == ttnn_predicted_labels[i]:
                dataset_ttnn_correct += 1
                correct += 1

        dataset_ttnn_accuracy = dataset_ttnn_correct / (batch_size)

        logger.info(
            f"ImageNet Inference Accuracy for iter {iters} of {batch_size} input samples : {dataset_ttnn_accuracy}"
        )

    accuracy = correct / (batch_size * iterations)
    logger.info(f"ImageNet Inference Accuracy for {batch_size}x{iterations} Samples : {accuracy}")
    assert accuracy >= 0.998, f"Expected accuracy : {0.998} Actual accuracy: {accuracy}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("iterations", [1])
def test_alexnet_on_mnist(device, batch_size, iterations):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_alexnet_on_mnist(device, batch_size, iterations)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_alexnet_on_imageFolder(device, batch_size):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_alexnet_on_imageFolder(device, batch_size)
