Quantization Guidelines & Calibration Checklist

Goal
Provide a repeatable workflow to convert a trained detection/classification model to TFLite INT8 suitable for TFLite‑Micro on ESP32‑S3, plus a short calibration checklist.

Overview
- Preferred flow: Train model in PyTorch or TensorFlow on desktop → convert to ONNX (if PyTorch) → convert to TFLite → post-training quantize (PTQ) to INT8 using a representative dataset → validate with TFLite interpreter on host → integrate into TFLite‑Micro build and test on hardware.
- If PTQ causes unacceptable accuracy loss, use Quantization-Aware Training (QAT) and then export/quantize.

Representative dataset (calibration)
- Size: ~200–500 images covering day/night, distances, occlusions, reflections, camera angles.
- Diversity: include small/distant lights, close frames, traffic lights with arrows, partial occlusions, and various backgrounds.
- Preprocessing: provide images in the same preprocessing as model expects (resize, normalize) or provide raw frames and let converter handle preprocessing.

PTQ steps (TensorFlow example)
1. Export model to TFLite (float32):
   - If using TensorFlow: use `tf.lite.TFLiteConverter.from_saved_model()` or from Keras model.
   - If using PyTorch: export to ONNX, then use `tf2onnx` or ONNX→TFLite intermediate tools.
2. Create a representative dataset generator function that yields preprocessed input arrays (numpy) for the converter.
3. Run converter with optimizations and representative dataset:
   - set `optimizations = [tf.lite.Optimize.DEFAULT]`
   - set representative dataset generator
   - enable `target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`
   - set `inference_input_type = tf.uint8 or tf.int8` and `inference_output_type = tf.uint8 or tf.int8` as needed.
4. Save `model_quant.tflite`.

QAT (if needed)
- If PTQ degrades accuracy too much, implement QAT in TensorFlow Model Optimization Toolkit or PyTorch QAT tools. Train for several epochs with fake-quant layers, then export and convert as above.

Post-conversion checks on host
- Run the TFLite interpreter (Python) and compute evaluation metrics on a validation set: mAP (detection), per-class accuracy (state), and any custom metrics.
- Measure latency with the TFLite interpreter (`interpreter.invoke()` timing) on host to estimate relative speed.
- Validate outputs match FP32 behavior reasonably. Examine confusion matrix for state labels.

TFLite‑Micro integration tips
- Use `xxd` or `incbin` to embed the `.tflite` model as a C array in the firmware (TFLite‑Micro expects model in binary).
- Configure the arena size conservatively; if you see `arena too small` errors, increase arena until fits, but keep memory footprint minimal.
- Avoid dynamic memory allocations; preallocate I/O and scratch buffers.

Calibration checklist (representative images)
- [ ] Collect 200–500 images for calibration with varied conditions:
  - [ ] Daylight, direct sun
  - [ ] Dawn/dusk
  - [ ] Night (streetlights present)
  - [ ] Rain/wet surfaces
  - [ ] Glare/reflection examples
  - [ ] Occlusion/partial views
  - [ ] Distant small lights and close-ups
  - [ ] Arrow signals and turn lamps
- [ ] Ensure images are representative of the target camera (same lens/field-of-view and resolution).
- [ ] Preprocess images identically to model input (resize + normalization) in the representative generator.
- [ ] Run PTQ workflow and save `model_quant.tflite`.
- [ ] Evaluate model_quant on host TFLite interpreter — record metric delta vs FP32.
- [ ] If delta acceptable, prepare TFLite‑Micro integration. If not, try QAT and repeat.
- [ ] Confirm model file size and memory arena requirements fit ESP32‑S3 resources.

Practical tips & pitfalls
- Avoid unsupported ops in TFLite‑Micro. If converter inserts `CUSTOM` ops, address them in model architecture or implement custom op in firmware.
- Use simple resize and normalization ops in the computational graph; prefer explicit preprocessing on device when possible.
- When using PyTorch→ONNX→TFLite, carefully check operator compatibility and numeric fidelity.
- If model uses non-linear postprocessing (NMS), implement NMS in firmware in C/C++ if not available in TFLite‑Micro.

Validation & regression
- Keep a small validation suite (50–200 images) separate from calibration images to measure real accuracy.
- Record baseline FP32 metrics and post-quant metrics to track regressions.

Deliverables from this process
- `model_quant.tflite` (INT8)
- Calibration dataset folder `data/representative_calib/` (200–500 images)
- Host validation script to run TFLite inference and output metrics
- Memory & latency report (estimated arena size, peak memory, inference ms)
