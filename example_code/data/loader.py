"""Dataset loader and COCO exporter (template)

Implements a minimal DatasetLoader that loads images and COCO-style
annotations with `attributes.state` (red/amber/green/off/arrow).

Fill in dataset paths and adapt transforms as needed.
"""
from pathlib import Path
from typing import List, Dict, Any
import json
import cv2

class DatasetLoader:
    def __init__(self, images_dir: str, ann_path: str = None):
        self.images_dir = Path(images_dir)
        self.ann_path = Path(ann_path) if ann_path else None
        self.images = sorted(self.images_dir.glob("*.jpg"))
        self.annotations = {}
        if self.ann_path and self.ann_path.exists():
            with open(self.ann_path, 'r', encoding='utf8') as f:
                self.coco = json.load(f)
        else:
            self.coco = None

    def get_image(self, idx: int):
        p = self.images[idx]
        img = cv2.imread(str(p))
        return img, p.name

    def export_coco(self, out_path: str, anns: List[Dict[str, Any]]):
        # Very small helper to write simple COCO file with `attributes.state`
        coco = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'traffic_light'}
            ]
        }
        img_id = 1
        ann_id = 1
        for p in self.images:
            coco['images'].append({'id': img_id, 'file_name': p.name, 'width': 0, 'height': 0})
            # user-provided anns should be matched to images by file_name
            for a in [x for x in anns if x.get('image_id') == p.name]:
                ann = {
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': 1,
                    'bbox': a['bbox'],
                    'area': a.get('area', a['bbox'][2]*a['bbox'][3]),
                    'iscrowd': 0,
                    'attributes': {'state': a.get('state', 'unknown'), 'visibility': a.get('visibility', 1.0)}
                }
                coco['annotations'].append(ann)
                ann_id += 1
            img_id += 1
        with open(out_path, 'w', encoding='utf8') as f:
            json.dump(coco, f, indent=2)

if __name__ == '__main__':
    print('Dataset loader template. Fill dataset paths and use DatasetLoader in scripts.')
