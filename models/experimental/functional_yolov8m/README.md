## Yolov8m Model

# Platforms:
    WH N150

## Introduction
YOLOv8 is the latest iteration in the YOLO series of real-time object detectors, offering cutting-edge performance in terms of accuracy and speed. Building upon the advancements of previous YOLO versions, YOLOv8 introduces new features and optimizations that make it an ideal choice for various object detection tasks in a wide range of applications.

# Details
The entry point to yolov8m model is YOLOv8m in `models/experimental/functional_yolov8m/tt/ttnn_yolov8m.py`. The model picks up weights from `yolov8m.pt` file located in `models/experimental/functional_yolov8m/demo/yolov8m.pt`. It is recommended to download the model weights from path `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt` inside the directory before running the tests.

**It is recommended to build the repository using this branch for proper model execution, as it might contain fixes that are not present in the v0.56.0 main branch.**

## Batch size: 8

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the `batch_size` to 8.

## Demo

Use `pytest --disable-warnings models/experimental/functional_yolov8m/demo/demo.py::test_demo` to run the yolov8m demo.

### Inputs

The demo receives inputs from `models/experimental/functional_yolov8m/demo/images` dir by default.

**Image Source: Google**

To test the model on different inputs:

1. Download test images and place them in the `models/experimental/functional_yolov8m/demo/images` directory.
2. Run the demo - `pytest --disable-warnings models/experimental/functional_yolov8m/demo/demo.py::test_demo`.

**Due to the large size of the KITTI dataset images, they are not included in the GitHub repository. However, testing has been performed using KITTI dataset images.**

### Outputs

The runs folder will be created inside the `models/experimental/functional_yolov8m/demo/` dir. For reference, the model output will be stored in `torch_model` dir, while the ttnn model output will be stored in `tt_model` dir.

## Performance Metrics
To generate and display performance statistics, run:
```sh
pytest models/experimental/functional_yolov8m/tests/test_yolov8m.py::test_perf_device_bare_metal_yolov8m
```

## Performance Profiling
Generate an operation-level performance sheet using:
```sh
./tt_metal/tools/profiler/profile_this.py -n <folder_name> -c "pytest tests/ttnn/integration_tests/yolov8m/test_ttnn_yolov8m.py::test_demo"
```
- **Performance Report Directory:** `generated/profiler/reports/<folder_name>`
