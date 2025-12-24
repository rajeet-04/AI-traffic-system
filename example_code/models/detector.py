"""Detector wrapper (template).

Provides a simple `Detector` class with a best-effort import of Ultralytics YOLO.
If YOLO isn't installed, `infer` returns an empty list — replace with a stub
or mock for offline dev.
"""
from typing import List, Dict, Any
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

class Detector:
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.model = None
        if _HAS_YOLO and model_path:
            self.model = YOLO(model_path)
            self.model.to(device)

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on a single BGR image.

        Returns list of detections: {'bbox':[x1,y1,x2,y2], 'label':state, 'score':float}
        """
        if self.model is None:
            # Fallback: no model available — return empty list
            return []
        results = self.model.predict(image, imgsz=640, conf=0.3, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                score = float(box.conf[0].cpu().numpy())
                # label mapping: assume model trained to output state as class names
                label_idx = int(box.cls[0].cpu().numpy())
                label = self.model.model.names.get(label_idx, str(label_idx)) if hasattr(self.model, 'model') else str(label_idx)
                detections.append({'bbox':[x1,y1,x2,y2], 'label': label, 'score': score})
        return detections

if __name__ == '__main__':
    print('Detector template. Provide a `model_path` trained for traffic light states.')
