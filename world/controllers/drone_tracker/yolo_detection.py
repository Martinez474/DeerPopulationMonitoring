"""Detect deer from Webots camera pixels, without simulator recognition labels."""

from pathlib import Path

from detection import DeerDetection, is_deer_model

MODEL_PATH = Path(__file__).resolve().parents[3] / "yolo_exported_models_and_weights/best.pt"


class YoloDeerDetector:
    mode = "yolo"

    def __init__(self, model_path=MODEL_PATH, confidence=0.35, image_size=640):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Deer model not found: {model_path}")
        import numpy as np
        from ultralytics import YOLO

        self.np = np
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.deer_classes = [index for index, name in self.model.names.items()
                             if is_deer_model(name)]
        if not self.deer_classes:
            raise ValueError(f"Model has no deer class: {self.model.names}")

    def detect(self, camera):
        image = camera.getImage()
        if image is None:
            return []  # First camera sample is not available yet.
        frame = self.np.frombuffer(image, dtype=self.np.uint8).reshape(
            camera.getHeight(), camera.getWidth(), 4
        )
        # Use only the first three channels (BGR) and convert to contiguous array for YOLO.
        bgr = self.np.ascontiguousarray(frame[:, :, :3])
        result = self.model.predict(
            source=bgr, imgsz=self.image_size, conf=self.confidence,
            classes=self.deer_classes, verbose=False,
        )[0]
        detections = []
        if result.boxes is None:
            return detections
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            if class_id not in self.deer_classes or score < self.confidence:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(DeerDetection(
                object_id=None, model=self.model.names[class_id],
                image_x=round((x1 + x2) / 2), image_y=round((y1 + y2) / 2),
                image_width=round(x2 - x1), image_height=round(y2 - y1),
                camera_x=None, camera_y=None, camera_z=None, confidence=score,
            ))
        return detections
