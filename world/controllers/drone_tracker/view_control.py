"""Keep the Webots 3D viewpoint on the drone's camera."""

import math
from typing import Any, Sequence


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


class DroneViewController:
    """Show only the onboard thermal or visible camera view."""

    def __init__(self, supervisor: Any, camera: Any, viewpoint_def: str):
        # connect the camera and world viewpoint
        self.camera = camera
        self.camera_node = supervisor.getFromDevice(camera._tag)
        self.viewpoint_node = supervisor.getFromDef(viewpoint_def)
        if self.camera_node is None:
            raise RuntimeError("Could not access the drone camera node")
        if self.viewpoint_node is None:
            raise RuntimeError(f"Could not find Viewpoint DEF {viewpoint_def!r}")

        # Remove Webots' follow smoothing and take direct control of the view.
        self.position = self.viewpoint_node.getField("position")
        self.orientation = self.viewpoint_node.getField("orientation")
        self.viewpoint_node.getField("follow").setSFString("")
        self.viewpoint_node.getField("followType").setSFString("None")
        self.viewpoint_node.getField("followSmoothness").setSFFloat(0.0)
        self.viewpoint_node.getField("near").setSFFloat(0.05)
        self.viewpoint_node.getField("fieldOfView").setSFFloat(camera.getFov())
        self.thermal_mode = True
        self.update()

    def update(self, position_correction=(0.0, 0.0, 0.0)) -> None:
        # Webots cameras look along local +X. The thermal screen sits behind
        # the sensing camera and between it and the 3D viewpoint.
        camera_position = self.camera_node.getPosition()
        camera_orientation = self.camera_node.getOrientation()
        optical_offset = -1.1 if self.thermal_mode else 0.8
        self.position.setSFVec3f([
            camera_position[0] + position_correction[0] + camera_orientation[0] * optical_offset,
            camera_position[1] + position_correction[1] + camera_orientation[3] * optical_offset,
            camera_position[2] + position_correction[2] + camera_orientation[6] * optical_offset,
        ])
        self.orientation.setSFRotation(rotation_matrix_to_axis_angle(camera_orientation))
