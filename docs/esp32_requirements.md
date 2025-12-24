ESP32‑S3 Requirements & Practical Notes

Purpose
This document lists recommended hardware, firmware/toolchain, and runtime constraints for running on-device traffic-signal detection and state classification on an ESP32‑S3 board with camera.

Target capture
- Resolution: capture at 640×480 @ 15–30 FPS (camera).
- Inference input: downscale to 320×240 (or 224×224) for model input to reduce compute.
- Capture format: MJPEG or JPEG frames for minimal decoding overhead.

Recommended hardware
- ESP32‑S3 development board with external PSRAM (strongly recommended).
  - PSRAM: the more, the better (4 MB+ recommended). If PSRAM is absent, expect strict model size limits.
- ESP32‑S3‑CAM modules (with PSRAM variant) or dev boards with OV2640/OV7670 camera connectors.
- Optional accelerators (if on-device strict real-time needed): Coral USB Accelerator or Raspberry Pi with Coral/EdgeTPU (not ESP32-only).

Flash & memory considerations
- Flash: >= 4 GB recommended for storing firmware, model artifacts, and logs during development.
- RAM/PSRAM: ESP32 RAM + PSRAM determine maximum model footprint. Aim for INT8 model ≤ 2 MB (ideally ≤ 1 MB) to fit comfortably.
- Stack/heap: reserve buffers for camera frame, intermediate tensors, and stack usage. TFLite‑Micro requires preallocated arena — tune arena size conservatively.

Model & inference targets
- Model size target (post-quantization INT8): ≤ 2 MB (target); ≤ 4 MB edge case if PSRAM available.
- Input size tradeoff: 320×240 gives better detection than 224×224 for small lights; try both.
- Throughput expectations: simple classifier or very tiny detector may reach several FPS; realistic all-in-one detection+state likely 1–5 FPS on ESP32‑S3.
- Prefer single-shot, low-op-count networks (MobileNetV3-small + SSD-like head, or custom tiny CNN detector) using simple ops supported by TFLite‑Micro.

Recommended software & toolchain
- Development: ESP‑IDF (latest stable) or Arduino-ESP32 (if you prefer Arduino flow). ESP-IDF gives more control.
- Runtime: TensorFlow Lite Micro (TFLite‑Micro) integrated into ESP‑IDF project.
- Build: use `idf.py` and CMake to include TFLite‑Micro and model binary.
- Host tooling: use TensorFlow and TFLite Converter on desktop to export and quantize models. Test with `tflite-runtime` or `tensorflow` TFLite interpreter on host before flashing.

Supported ops & portability
- Stick to standard, well-supported ops in TFLite (Conv2D, DepthwiseConv2D, FullyConnected, Reshape, Relu/Relu6, Add, MaxPool, Softmax).
- Avoid complex/unsupported ops (custom layers, dynamic resizing layers). If required, implement custom ops in TFLite‑Micro (complex and slow).

Runtime optimizations
- Use INT8 quantization (post-training quantization with representative dataset).
- Use fixed-size tensors and preallocate scratch/arena memory in TFLite‑Micro to avoid heap fragmentation.
- Crop/ROI: restrict processing to upper third of image or synthetic ROIs to reduce scanning area.
- Frame skipping: run the detector every N frames (e.g., every 2–3 frames) and run a lightweight classifier or smoothing in between.
- Minimal NMS: implement a simple NMS with small window to reduce detections.

Data & representative sampling
- Collect representative calibration images across: day/night, direct sun, backlight, rain, reflections, partial occlusion, different distances and camera angles.
- For quantization calibration, collect 200–500 representative images (diverse lighting/angles) in JPEG format.

Energy & power
- Consider power consumption: camera capture + inference may draw significant current. Use stable power supply during tests.

Debugging tips
- Test the TFLite model on a desktop TFLite interpreter with the representative dataset first.
- Use serial logging and UART to stream metrics (latency, memory use) from the device.
- Incrementally build capability: start with a tiny classifier (detect presence of lamp cluster) then move to detection.

Safety note
- This prototype is for research and prototyping only. Do not connect outputs to vehicle actuators without rigorous validation and safety checks.
