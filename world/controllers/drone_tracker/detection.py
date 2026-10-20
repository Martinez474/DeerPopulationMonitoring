"""Deer detection backends for the survey drone.

The visual backend uses Webots' camera-recognition output.  Keeping detection
behind this small interface lets a thermal backend replace it later without
changing flight control or the HUD.
"""

from dataclasses import asdict, dataclass
import re
from typing import Any


VISUAL_SHAPE_MODE = "visual_shape"


def is_deer_model(model: str) -> bool:
    """Return whether a Webots recognition model is labelled as a deer."""

    words = re.findall(r"[a-z0-9]+", (model or "").lower())
    return "deer" in words


@dataclass(frozen=True)
class DeerDetection:
    """The useful portion of one recognition result."""

    object_id: int
    model: str
    image_x: int
    image_y: int
    image_width: int
    image_height: int
    camera_x: float
    camera_y: float
    camera_z: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VisualShapeDeerDetector:
    """Count visible deer from Webots recognition silhouettes and labels."""

    mode = VISUAL_SHAPE_MODE

    def detect(self, camera: Any) -> list[DeerDetection]:
        detections: list[DeerDetection] = []

        # check every object found by the camera
        for recognized_object in camera.getRecognitionObjects():
            model = recognized_object.getModel()
            if not is_deer_model(model):
                continue

            # keep the data needed by the drone
            image_position = recognized_object.getPositionOnImage()
            image_size = recognized_object.getSizeOnImage()
            camera_position = recognized_object.getPosition()
            detections.append(
                DeerDetection(
                    object_id=recognized_object.getId(),
                    model=model,
                    image_x=int(image_position[0]),
                    image_y=int(image_position[1]),
                    image_width=int(image_size[0]),
                    image_height=int(image_size[1]),
                    camera_x=float(camera_position[0]),
                    camera_y=float(camera_position[1]),
                    camera_z=float(camera_position[2]),
                )
            )

        return detections
