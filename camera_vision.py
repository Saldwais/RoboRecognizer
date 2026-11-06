"""Local object recognition for RoboPilot's existing camera frames."""

from base64 import b64encode
from collections import Counter
from threading import Lock


class CameraVision:
    def __init__(self):
        self.lock = Lock()
        self.default_model = None
        self.custom_model = None
        self.custom_labels = ()

    def analyze(self, jpeg, targets, describe):
        labels = tuple(dict.fromkeys(word.strip() for word in targets.split(',') if word.strip()))
        if len(targets) > 200 or len(labels) > 8:
            raise ValueError('Enter up to eight objects and 200 characters total.')
        if not self.lock.acquire(blocking=False):
            return None
        try:
            return self._analyze(jpeg, labels, describe)
        finally:
            self.lock.release()

    def _analyze(self, jpeg, labels, describe):
        import cv2
        import numpy as np
        from ultralytics import YOLO, YOLOWorld

        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Could not decode camera frame.')
        if labels:
            if self.custom_model is None:
                self.custom_model = YOLOWorld('yolov8s-worldv2.pt')
            if labels != self.custom_labels:
                self.custom_model.set_classes(list(labels))
                self.custom_labels = labels
            model = self.custom_model
        else:
            if self.default_model is None:
                self.default_model = YOLO('yolo11n.pt')
            model = self.default_model

        result = model.predict(frame, imgsz=416, conf=0.35, max_det=25, device='cpu', verbose=False)[0]
        boxes = [
            {'label': result.names[int(class_id)], 'confidence': round(confidence, 2),
             'box': [round(value, 4) for value in bounds]}
            for class_id, confidence, bounds in zip(
                result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxyn.tolist())
        ]
        annotated = None
        if boxes:
            ok, encoded = cv2.imencode('.jpg', result.plot())
            if ok:
                annotated = b64encode(encoded).decode('ascii')
        description = ''
        if describe:
            counts = Counter(item['label'] for item in boxes)
            description = ('In this view: ' + ', '.join(f'{count} × {label}' for label, count in counts.items())
                           if counts else 'No matching objects detected in this frame.')
        return {'boxes': boxes, 'description': description, 'annotated': annotated}
