"""Switch the Webots 3D view between chase and onboard camera views."""

from dataclasses import dataclass
import math
from typing import Any, Sequence


CHASE_VIEW = "chase"
DRONE_VIEW = "drone"


def rotation_matrix_to_axis_angle(matrix: Sequence[float]) -> list[float]:
    """Convert a row-major 3x3 rotation matrix to Webots axis-angle form."""

    if len(matrix) != 9:
        raise ValueError("A 3x3 rotation matrix must contain nine values")

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix
    trace = m00 + m11 + m22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(max(0.0, 1.0 + m00 - m11 - m22)) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(max(0.0, 1.0 + m11 - m00 - m22)) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(max(0.0, 1.0 + m22 - m00 - m11)) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale

    length = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if length < 1e-12:
        return [0.0, 0.0, 1.0, 0.0]
    qw, qx, qy, qz = (value / length for value in (qw, qx, qy, qz))

    # keep the angle at most pi
    if qw < 0.0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz

    qw = max(-1.0, min(1.0, qw))
    angle = 2.0 * math.acos(qw)
    sine = math.sqrt(max(0.0, 1.0 - qw * qw))
    if sine < 1e-7:
        return [0.0, 0.0, 1.0, 0.0]
    return [qx / sine, qy / sine, qz / sine, angle]


@dataclass
class ViewSnapshot:
    position: list[float]
    orientation: list[float]
    field_of_view: float
    follow: str
    follow_type: str
    follow_smoothness: float


class DroneViewController:
    """Own the chase/onboard view toggle for a supervisor controller."""

    def __init__(self, supervisor: Any, camera: Any, viewpoint_def: str):
        # connect the camera and world viewpoint
        self.camera = camera
        self.camera_node = supervisor.getFromDevice(camera._tag)
        self.viewpoint_node = supervisor.getFromDef(viewpoint_def)
        if self.camera_node is None:
            raise RuntimeError("Could not access the drone camera node")
        if self.viewpoint_node is None:
            raise RuntimeError(f"Could not find Viewpoint DEF {viewpoint_def!r}")

        # keep access to the viewpoint fields
        self.position = self.viewpoint_node.getField("position")
        self.orientation = self.viewpoint_node.getField("orientation")
        self.field_of_view = self.viewpoint_node.getField("fieldOfView")
        self.follow = self.viewpoint_node.getField("follow")
        self.follow_type = self.viewpoint_node.getField("followType")
        self.follow_smoothness = self.viewpoint_node.getField("followSmoothness")
        # remember the original chase view
        self.mode = CHASE_VIEW
        self.chase_snapshot = self._snapshot()

    @property
    def label(self) -> str:
        return "DRONE VIEW" if self.mode == DRONE_VIEW else "CHASE VIEW"

    def _snapshot(self) -> ViewSnapshot:
        return ViewSnapshot(
            position=list(self.position.getSFVec3f()),
            orientation=list(self.orientation.getSFRotation()),
            field_of_view=self.field_of_view.getSFFloat(),
            follow=self.follow.getSFString(),
            follow_type=self.follow_type.getSFString(),
            follow_smoothness=self.follow_smoothness.getSFFloat(),
        )

    def toggle(self) -> str:
        if self.mode == CHASE_VIEW:
            # move the viewpoint onto the drone camera
            self.chase_snapshot = self._snapshot()
            self.mode = DRONE_VIEW
            self.follow.setSFString("")
            self.follow_type.setSFString("None")
            self.field_of_view.setSFFloat(self.camera.getFov())
            self.update()
        else:
            # restore the saved chase view
            self.mode = CHASE_VIEW
            snapshot = self.chase_snapshot
            self.position.setSFVec3f(snapshot.position)
            self.orientation.setSFRotation(snapshot.orientation)
            self.field_of_view.setSFFloat(snapshot.field_of_view)
            self.follow.setSFString(snapshot.follow)
            self.follow_type.setSFString(snapshot.follow_type)
            self.follow_smoothness.setSFFloat(snapshot.follow_smoothness)
        return self.mode

    def update(self) -> None:
        if self.mode != DRONE_VIEW:
            return

        # match the viewpoint to the camera pose
        self.position.setSFVec3f(list(self.camera_node.getPosition()))
        self.orientation.setSFRotation(
            rotation_matrix_to_axis_angle(self.camera_node.getOrientation())
        )
