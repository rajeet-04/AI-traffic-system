"""Demo server: accepts JPEG frame POSTs and runs detection->tracking->decision.

ESP32-S3 can POST frames as multipart/form-data to /frame. The server returns
JSON with advisory and a small overlay image (optional).

Usage:
    python scripts/run_demo.py

Dependencies: Flask, OpenCV, numpy
"""
from flask import Flask, request, jsonify, send_file
import io
import cv2
import numpy as np
from PIL import Image
import threading
import time
from models.detector import Detector
from modules.tracker import SimpleTracker
from controllers.decision import DecisionEngine

app = Flask(__name__)

# Initialize components (model_path optional)
DETECTOR = Detector(model_path=None)  # set model_path to a trained YOLO file
TRACKER = SimpleTracker(max_distance=80.0, max_age=8)
DECISION = DecisionEngine(window=5, red_threshold=0.6)

lock = threading.Lock()
last_result = None

def draw_overlay(image: np.ndarray, tracks):
    img = image.copy()
    for t in tracks:
        x1,y1,x2,y2 = map(int, t.bbox)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, f'ID:{t.id} {t.label}', (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img

@app.route('/frame', methods=['POST'])
def receive_frame():
    """Accepts `multipart/form-data` with file form field `frame` (jpeg).
    Returns JSON: advisory + per-track info. Optionally returns overlay image when
    `?overlay=1` is passed.
    """
    global last_result
    if 'frame' not in request.files:
        return jsonify({'error': 'no frame field'}), 400
    file = request.files['frame']
    buf = file.read()
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'could not decode image'}), 400

    # inference
    detections = DETECTOR.infer(img)
    tracks = TRACKER.update(detections)
    decision = DECISION.decide(tracks)

    # overlay image if requested
    overlay_bytes = None
    if request.args.get('overlay') == '1':
        over = draw_overlay(img, tracks)
        _, imbuf = cv2.imencode('.jpg', over)
        overlay_bytes = imbuf.tobytes()

    with lock:
        last_result = {'time': time.time(), 'decision': decision}

    response = {'decision': decision}
    if overlay_bytes:
        return send_file(io.BytesIO(overlay_bytes), mimetype='image/jpeg')
    return jsonify(response)

@app.route('/status', methods=['GET'])
def status():
    with lock:
        return jsonify(last_result or {'time': None})

if __name__ == '__main__':
    # Run in threaded mode to accept multiple incoming frames
    app.run(host='0.0.0.0', port=5000, threaded=True)
