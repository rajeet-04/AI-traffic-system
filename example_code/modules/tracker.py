"""Lightweight tracker template.

Simple centroid-based tracker that assigns incremental IDs and keeps a short
history of detections for each track. Replace with ByteTrack/DeepSORT for
production-quality tracking.
"""
from typing import List, Dict, Any, Tuple
import numpy as np

class Track:
    def __init__(self, tid: int, bbox: List[float], label: str, score: float):
        self.id = tid
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.label = label
        self.score = score
        self.history = [label]
        self.age = 0

    def update(self, bbox: List[float], label: str, score: float):
        self.bbox = bbox
        self.label = label
        self.score = score
        self.history.append(label)
        if len(self.history) > 10:
            self.history.pop(0)
        self.age = 0

class SimpleTracker:
    def __init__(self, max_distance: float = 50.0, max_age: int = 5):
        self.max_distance = max_distance
        self.max_age = max_age
        self.tracks = {}
        self._next_id = 1

    def _centroid(self, bbox: List[float]) -> Tuple[float,float]:
        x1,y1,x2,y2 = bbox
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def update(self, detections: List[Dict[str,Any]]) -> List[Track]:
        # detections: list of {'bbox':[x1,y1,x2,y2],'label':str,'score':float}
        assigned = set()
        det_centroids = [self._centroid(d['bbox']) for d in detections]
        det_assigned = [None]*len(detections)

        # compute distance from existing tracks
        track_items = list(self.tracks.items())
        track_centroids = [self._centroid(t.bbox) for _,t in track_items]
        for i, det_c in enumerate(det_centroids):
            best_j = None
            best_d = float('inf')
            for j, (tid, track) in enumerate(track_items):
                if j in assigned:
                    continue
                tc = track_centroids[j]
                d = np.hypot(det_c[0]-tc[0], det_c[1]-tc[1])
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j is not None and best_d <= self.max_distance:
                tid, track = track_items[best_j]
                track.update(detections[i]['bbox'], detections[i]['label'], detections[i]['score'])
                det_assigned[i] = tid
                assigned.add(best_j)

        # create new tracks for unassigned detections
        for i, d in enumerate(detections):
            if det_assigned[i] is None:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, d['bbox'], d['label'], d['score'])

        # age and remove old tracks
        to_delete = []
        for tid, track in list(self.tracks.items()):
            track.age += 1
            if track.age > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return list(self.tracks.values())

if __name__ == '__main__':
    print('Simple tracker template. Replace with DeepSORT/ByteTrack for better performance.')
