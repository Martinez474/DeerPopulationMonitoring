"""Survey clock and ground-truth checks for the simulated thermal sensor.

Deer scene nodes are used here only to evaluate image-derived candidates.
The active thermal detector in thermal_image_detection.py never reads them.
"""

import math
from typing import Sequence

from detection import DeerDetection


DAY_SECONDS = 600.0
START_HOUR = 6.0
DEER_DEFS = tuple(f"THERMAL_DEER_{number}" for number in range(1, 11))


def hour_of_day(simulation_time: float, offset_hours: float = 0.0) -> float:
    return (START_HOUR + simulation_time * 24.0 / DAY_SECONDS + offset_hours) % 24.0


def background_temperature_c(hour: float) -> float:
    """Air/terrain reference warms during the day and cools overnight."""

    return 19.0 + 8.0 * math.cos(2.0 * math.pi * (hour - 15.0) / 24.0)


def sunlight_fraction(hour: float) -> float:
    """Daylight rises at 06:00, peaks at noon, and ends at 18:00."""

    return max(0.0, math.sin(math.pi * (hour - 6.0) / 12.0))


def sunlight_direction(hour: float) -> list[float]:
    """Direction of parallel light rays in this world's Z-up coordinates."""

    phase = math.pi * (hour - 6.0) / 12.0
    elevation = math.radians(5.0 + 65.0 * sunlight_fraction(hour))
    horizontal = math.cos(elevation)
    return [
        -horizontal * math.cos(phase),
        horizontal * math.sin(phase),
        -math.sin(elevation),
    ]


def project_to_camera(camera_position, orientation, target, width, height, fov):
    """Project a world point through Webots' local +X camera axis."""

    delta = [target[i] - camera_position[i] for i in range(3)]
    depth = sum(orientation[i * 3] * delta[i] for i in range(3))
    right = -sum(orientation[i * 3 + 1] * delta[i] for i in range(3))
    up = sum(orientation[i * 3 + 2] * delta[i] for i in range(3))
    if depth <= 0.2 or depth > 250.0:
        return None
    focal = width / (2.0 * math.tan(fov / 2.0))
    x = width / 2.0 + focal * right / depth
    y = height / 2.0 - focal * up / depth
    if not (0.0 <= x < width and 0.0 <= y < height):
        return None
    return x, y, depth, focal


def count_ground_truth_matches(
    candidates: Sequence[DeerDetection], deer_positions: Sequence[Sequence[float]],
    camera_position, camera_orientation, width: int, height: int, fov: float,
) -> int:
    """Count nearby known deer solely for reporting detector accuracy."""

    projected_deer = []
    for deer_position in deer_positions:
        target = (deer_position[0], deer_position[1], deer_position[2] + 1.1)
        projection = project_to_camera(camera_position, camera_orientation,
                                       target, width, height, fov)
        if projection is not None:
            projected_deer.append(projection)

    matches = 0
    for candidate in candidates:
        allowed = max(24.0, 0.75 * max(candidate.image_width, candidate.image_height))
        nearest = min(
            range(len(projected_deer)),
            key=lambda index: math.hypot(candidate.image_x - projected_deer[index][0],
                                         candidate.image_y - projected_deer[index][1]),
            default=None,
        )
        if nearest is not None:
            deer = projected_deer[nearest]
            if math.hypot(candidate.image_x - deer[0], candidate.image_y - deer[1]) <= allowed:
                matches += 1
                projected_deer.pop(nearest)
    return matches
