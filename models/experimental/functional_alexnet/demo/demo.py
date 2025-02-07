# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import random
import pytest
import torch, ttnn
from PIL import Image
from loguru import logger
from torchvision import models, transforms
from models.experimental.functional_alexnet.tt.ttnn_alexnet import ttnn_alexnet
from models.utility_functions import disable_persistent_kernel_cache, disable_compilation_reports
from models.experimental.functional_alexnet.tt.ttnn_alexnet_utils import custom_preprocessor


def get_dataset(batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize(224),
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

    state_dict = torch_model.state_dict()
    parameters = custom_preprocessor(device, state_dict=state_dict)

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


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_alexnet_on_imageFolder(device, batch_size):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_alexnet_on_imageFolder(device, batch_size)
