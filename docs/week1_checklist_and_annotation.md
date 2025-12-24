Week‑1 Detailed Checklist & Annotation Template

Objective (Week 1)
Assemble a small, high-quality dataset suitable for training and quantization calibration, and establish annotation standards to ensure consistent labels for traffic lights and states.

Deliverables (end of Week 1)
- `data/representative_calib/` with 200–500 calibration images
- `data/train/` and `data/val/` subsets (COCO JSON) with at least 500 annotated images total
- `data/loader.py` ready to read COCO-style annotations with `attributes.state`
- Annotation guideline document + QC checklist

Checklist
1. Capture & collect images
   - [ ] Mount ESP32‑S3 camera and capture sample video and stills at 640×480, 15–30 FPS.
   - [ ] Collect images across conditions:
     - Daylight (various angles)
     - Dawn/dusk
     - Night (street lighting)
     - Rain/wet road reflections
     - Backlit and glare
     - Partial occlusions (trees, poles)
     - Different distances (near, mid, far)
   - Target counts: aim for 500–1500 images total; at minimum 500 for prototype.

2. Select calibration images
   - [ ] From collected images, choose 200–500 representative images for quantization calibration (diverse lighting and distances).

3. Annotation conventions (COCO JSON)
   - Categories:
     - `traffic_light` as primary category (category_id = 1)
   - State labels: store as attribute `state` in each annotation or encode as sub-category ids.
     - Allowed states: `red`, `amber` (or `yellow`), `green`, `off`, `arrow_red`, `arrow_green`, `unknown`.
   - Annotation fields example (COCO annotation entry):
     {
       "id": 1,
       "image_id": 1,
       "category_id": 1,
       "bbox": [x, y, width, height],
       "area": width*height,
       "iscrowd": 0,
       "attributes": {"state": "red", "visibility": 0.9}
     }
   - Bounding box rules:
     - Box must tightly enclose the lamp cluster (all visible lights in the fixture).
     - For arrow signals, include the arrow lamp region.
     - For stacked lights (vertical/horizontal), include entire assembly.
   - Visibility & occlusion:
     - `visibility`: float 0.0–1.0 estimating visible fraction.
     - Mark `occluded` or partial if more than 30% occluded.

4. Annotation tool & workflow
   - Recommended tools: CVAT (video support + attributes), LabelImg for single images, LabelStudio for collaborative labeling.
   - Use attribute field for `state` to allow multi-class state per bbox.
   - QC workflow:
     - Annotator A labels batch
     - Annotator B reviews 10% randomly for errors
     - Resolve conflicts; record inter-annotator agreement

5. Train/val split
   - Split by video/scene to avoid frame leakage.
   - Typical split: 80% train, 20% val by scene.
   - Reserve separate `data/test/` from different locations/lighting for final evaluation.

6. Export & sanity checks
   - [ ] Export COCO JSON with `images`, `annotations`, `categories` and `attributes` as required.
   - [ ] Run a sanity script to visualize 50 random annotated images and check label correctness.
   - [ ] Ensure no duplicate `image_id` collisions and all `image_id` fields match file names.

Annotation examples & edge cases
- Small distant lights: If lamp cluster is <10 px high at 640×480, annotate but flag as `tiny=true` in attributes if you want to evaluate AP_small separately.
- Reflections: Do not annotate reflections as lights unless it is an independent physical lamp.
- Multiple lights in frame: annotate each fixture separately with its own state.
- Traffic sign lamps (not traffic signals): do not annotate unless they function as traffic-light signals.

Labeling speed & quality tips
- Use hotkeys and attribute presets in CVAT to speed up state labeling.
- Provide an examples sheet to annotators showing clear red/amber/green/off/arrow instances.
- Encourage annotators to mark ambiguous cases as `unknown` rather than guessing.

QC checklist (before Week 2 training)
- [ ] At least 500 images annotated with `state` attributes.
- [ ] 200–500 calibration images chosen and separated.
- [ ] Visual inspection of 100 random annotations passed.
- [ ] Train/val/test splits exported and verified.
- [ ] Representative distribution: day/night/occlusions present in both train and val.

Notes for training team
- Preprocess training images to match on-device input pipeline (resize to model input, normalization).
- Keep a small evaluation harness to check TFLite INT8 outputs vs FP32 during quantization.

Next steps after Week 1
- Start training tiny detector on desktop using the annotated dataset and calibration images for PTQ.
- Iterate with quantization and evaluate on the held-out test set.
