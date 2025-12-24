Plan: Traffic Signal & Light Detection Prototype

TL;DR
Build a modular sense → decide → act prototype that detects traffic lights, classifies their state, tracks lights across frames, and produces safe advisories. Start local for development, then target constrained edge (ESP32‑S3) using the ESP32 as a camera + offload pattern (recommended for 4‑week delivery).

Goals
- Teach ADAS perception & decision-making (student-friendly).
- Produce a working research prototype, demo video, and paper-ready results.
- Integrate XAI and multi-agent coordination hooks later.

MVP Definition
- Per-frame bounding boxes + state labels (red/amber/green/off/arrow).
- Persistent track IDs across frames, simple temporal smoothing of state.
- Rule-based decision module that outputs STOP/GO/CAUTION advisories (log-only initially).
- Desktop-GPU prototype first; ESP32‑S3 used as camera and streamer with inference offloaded to a Pi/PC.

Architecture (modules & flow)
- Input: camera/video stream
- Preprocess: resize, normalize, optional ROI / tiling
- Detector: single-frame detector -> (bbox, state, score)
- Classifier (opt): per-crop state classifier if detector state is weak
- Tracker: associate detections -> persistent tracks with history
- Decision: FSM (rule-based) consuming track history -> advisory
- Logger/Eval: store detections, GT, latencies
- Visualizer/Demo: overlay results and export video
Dataflow: Input → Preprocess → Detector → Tracker → Decision → Logger/Visualizer

Per-module technology choices
- Detector: YOLOv8 (Ultralytics) for iteration and export; fallback Detectron2 / Faster R-CNN for accuracy.
- State classification: multi-head detector or small classifier (ResNet18/MobileNetV3).
- Tracker: ByteTrack or DeepSORT; SORT as lightweight fallback.
- Augmentation: Albumentations (brightness, blur, cutout, random crop, night augmentation).
- Training: PyTorch; logging with TensorBoard or Weights & Biases.
- Inference export: ONNX → TensorRT / TFLite (for Coral/ESP32 S3) / Edge TPU compile.

Datasets (start with these)
- Bosch Small Traffic Lights: https://hci.iwr.uni-heidelberg.de/node/613
- BDD100K: https://bdd-data.berkeley.edu/
- Mapillary Vistas: https://www.mapillary.com/datasets
- COCO (pretrain): https://cocodataset.org/
- Cityscapes: https://www.cityscapes-dataset.com/
- Udacity Traffic Light (examples): https://github.com/udacity/CarND-Capstone

Annotation & formats
- Tools: CVAT (video + attributes), LabelImg, LabelStudio.
- Format: COCO JSON preferred; use `annotations[].attributes.state` or distinct category ids per state. Keep visibility/occlusion field and frame index.

Metrics & evaluation
- Detection: mAP@[.50:.95], mAP@0.5, AP_small.
- State: per-class accuracy, F1, confusion matrix.
- Tracking: MOTA, ID switches, MOTP.
- Decision: scenario success rate (correct advisory), latency-to-decision.
- Procedure: split by videos (no frame leakage), hold-out weather/lighting splits (night, rain).

Real-time & latency
- Target: 10–20 FPS for prototype; prefer detector <50 ms on desktop GPU.
- Optimizations: FP16/INT8 quantization, TensorRT, asynchronous pipeline, ROI/tiling for small objects.

Hardware & edge patterns (ESP32‑S3 focus)
- Development: desktop GPU (RTX 20/30 series) for training and debug.
- Edge options and recommended path for 4 weeks:
  - Recommended: ESP32‑S3 as camera + uploader (MJPEG) -> offload inference to a Raspberry Pi 4 or PC. This avoids memory limitations and meets the deadline.
  - Ambitious: attempt on-device TFLite-micro on ESP32‑S3 — likely infeasible for non-trivial detectors within 4 weeks due to RAM/model size limits.
  - Better-capable edge: Raspberry Pi 4 / Coral USB Accelerator / Jetson Nano if local inference is required.
- ESP32 notes: memory/CPU limited; use MJPEG stream + server-side inference or pair with Pi/Coral for on-device inference.

Safety, failure modes & mitigations
- Failure modes: misclassify amber/red, miss small/distant lights, false positives from reflections.
- Mitigations: temporal voting (window), conservative thresholds (prefer safe advisories), ensemble (detector + classifier), human-in-loop for actuation.
- Ethics: do not deploy actuation from prototype without full validation; respect dataset licenses.

Risks & mitigation
- Dataset bias (night/occlusion): collect targeted data and augment.
- Edge performance shortfall: plan quantization and tiny-model fallbacks.
- Legal/ethical: verify dataset licenses and include safety disclaimers.

4‑Week Schedule (deadline: 2026-01-21)
Week 1 — Data & baseline
- Goals: assemble a small curated dataset (LISA + Bosch samples), agree annotation format, implement `data/loader.py`.
- Deliverables:
  - `data/loader.py` (COCO-style loader that supports `attributes.state` and `visibility`)
  - `data/prepare_dataset.py` (convert annotations; sample subset)
  - small validation set (day/night/occlusion)
- Tasks:
  - Camera resolution target: recommend 640×480 @ 15–30 FPS for ESP32‑S3.
  - Annotate ~500–1500 images (prioritize small/distant lights + night).

Week 2 — Train & validate detector on desktop
- Goals: train a lightweight detector and per-crop classifier; iterate model/augmentation.
- Model choices:
  - Tiny option A (preferred): MobileNetV3 backbone + SSD / YOLO-nano / YOLOv8n (small)
  - Option B: Pruned YOLOv8-small if more accuracy needed
- Deliverables:
  - `models/detector.py` (train/infer wrapper)
  - initial eval `eval/evaluate.py` (mAP@0.5, state accuracy)
- Tasks:
  - Use Albumentations (brightness, blur, night augment) and class balancing.
  - Aim for detector inference <100ms on desktop for faster iterations.

Week 3 — Quantize & edge-test
- Goals: export to TFLite, quantize (INT8), test on host; prepare ESP32‑S3 compatibility plan.
- Deliverables:
  - `scripts/export_tflite.py` and `scripts/quantize.py`
  - host-side TFLite test harness `scripts/run_tflite_host.py`
  - notes for ESP32-S3 memory limits and model size targets (<2–4 MB ideal)
- Tasks & options:
  - Option 1 (recommended short path): keep ESP32‑S3 as camera + stream MJPEG frames to a Pi/PC for inference.
  - Option 2 (ambitious): TFLite-micro on ESP32‑S3 — requires model small enough and special build (use ESP-IDF + TinyML examples). Expect accuracy drop.
  - Test INT8 accuracy vs FP32; prefer post-training quantization with a representative dataset.

Week 4 — Integration & demo
- Goals: integrate camera streaming, inference path, tracker, FSM decision, and produce demo video + docs.
- Deliverables:
  - `modules/tracker.py` (lightweight SORT/byte-style)
  - `controllers/decision.py` (windowed voting FSM; outputs advisories/log)
  - `scripts/run_demo.py` (ESP32 stream input → inference → tracker → decision → overlay)
  - `README.md` update with deployment instructions for ESP32‑S3 (+ optional Pi/Coral)
- Tasks:
  - If using offload: implement MJPEG streamer on ESP32‑S3 and a small Flask/fastAPI receiver for inference.
  - If attempting on-device: flash test firmware and run small classifier (expect significant tuning).

Concrete file list to produce
- `data/loader.py` — COCO loader with `state` attribute
- `data/prepare_dataset.py` — sample subset exporter
- `models/detector.py` — wrapper to train and run (YOLOv8n/MobileNetV3-SSD)
- `scripts/export_tflite.py` — export & quantize
- `scripts/run_demo.py` — server that accepts MJPEG frames and runs inference
- `modules/tracker.py`, `controllers/decision.py`, `eval/evaluate.py`, `requirements.txt`

Minimal runnable prototype (first tasks)
1. `data/loader.py` — implement loader for a small subset (LISA/Bosch), with COCO export script.
2. `models/detector.py` — load pretrained YOLOv8, run inference on single frames and output state.
3. `modules/tracker.py` — integrate ByteTrack/DeepSORT and maintain track history.
4. `controllers/decision.py` — conservative FSM that aggregates last N states to output advisory.
5. `scripts/run_demo.py` — assemble pipeline and save overlay video + logs.

Immediate next actions (choose one)
A) Generate the repo file templates and `requirements.txt` now.
B) Produce the detailed Week‑1 checklist and an annotation guideline template.
C) Generate the ESP32‑S3 streamer + server demo code (`scripts/run_demo.py`) that accepts MJPEG and runs the desktop model.

Quick confirmations needed
- Confirm camera target: `640x480 @ 15-30 FPS` (yes/no or give other).
- Confirm inference mode: `ESP32‑S3 as camera + offload to Pi/PC` (recommended) or `attempt on-device TFLite-micro` (risky in 4 weeks).

Notes
- This plan focuses on delivering a working prototype by 2026-01-21 using ESP32‑S3 as a camera + offload. If you prefer attempting on-device inference on the ESP32‑S3, expect extra time for TinyML integration and aggressive model shrinking.

--
Created for iterative refinement. If you want, I can now create the repo templates and `requirements.txt` (option A) and scaffold the demo server for the ESP32 stream (option C).
