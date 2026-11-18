"""Build an apparent-temperature image from the surfaces visible to the drone.

The RangeFinder supplies the first visible surface at each pixel.  Webots'
recognition segmentation supplies the visible deer silhouette.  The remaining
surfaces in this survey world are terrain, three shrub canopies, and sky.
Temperatures and emissivities are illustrative parameters, not measurements.
"""

from dataclasses import dataclass
import math
from typing import Sequence

from thermal_detection import background_temperature_c, sunlight_fraction

try:
    import numpy as np
except ImportError:
    np = None


SKY, TERRAIN, SHRUB, DEER = range(4)
EMISSIVITY = (1.0, 0.96, 0.98, 0.97)
SHRUB_RADIUS_XY = 2.6
SHRUB_RADIUS_Z = 2.2


@dataclass
class ThermalFrame:
    width: int
    height: int
    temperatures: list[float]
    surfaces: bytearray
    background_c: float


def ray_ellipsoid_depth(origin, direction, center, horizontal_radius, vertical_radius):
    """Return the entry depth along a camera ray, in optical-axis metres."""

    ox = (origin[0] - center[0]) / horizontal_radius
    oy = (origin[1] - center[1]) / horizontal_radius
    oz = (origin[2] - center[2]) / vertical_radius
    dx = direction[0] / horizontal_radius
    dy = direction[1] / horizontal_radius
    dz = direction[2] / vertical_radius
    a = dx * dx + dy * dy + dz * dz
    b = 2.0 * (ox * dx + oy * dy + oz * dz)
    c = ox * ox + oy * oy + oz * oz - 1.0
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    root = math.sqrt(discriminant)
    near = (-b - root) / (2.0 * a)
    far = (-b + root) / (2.0 * a)
    if far <= 0.0:
        return None
    return max(0.0, near)


def surface_temperature_c(surface: int, world_point, sun: float, ambient_c: float) -> float:
    """Simple surface energy model with different heating and cooling rates."""

    x, y, z = world_point
    if surface == SKY:
        return ambient_c - 16.0
    if surface == DEER:
        # Fur is cooler than core body temperature, and has mild texture.
        return 27.5 + 0.12 * (ambient_c - 19.0) + 0.6 * sun + 0.35 * math.sin(x * 3.1 + y * 2.3 + z)
    if surface == SHRUB:
        # Transpiring leaves stay closer to air temperature than sunlit soil.
        return ambient_c - 1.2 + 2.0 * sun + 0.5 * math.sin(x * 0.7 + y * 0.4)
    # Terrain has greater solar gain and retains some warmth after sunset.
    solar_gain = 5.5 * sun
    texture = 0.9 * math.sin(x * 0.045) * math.cos(y * 0.055)
    altitude_cooling = 0.006 * (z - 200.0)
    return ambient_c - 0.8 + solar_gain + texture - altitude_cooling


def apparent_temperature_c(surface_c: float, emissivity: float, reflected_c: float,
                           air_c: float, distance_m: float) -> float:
    """Approximate emitted/reflected radiance and atmospheric loss using T^4."""

    kelvin = surface_c + 273.15
    reflected = reflected_c + 273.15
    air = air_c + 273.15
    transmission = math.exp(-0.00018 * distance_m)
    radiance = transmission * (emissivity * kelvin ** 4 + (1.0 - emissivity) * reflected ** 4)
    radiance += (1.0 - transmission) * air ** 4
    return radiance ** 0.25 - 273.15


def render_temperature_frame(
    depths: Sequence[float],
    segmentation: bytes | None,
    camera_position: Sequence[float],
    camera_orientation: Sequence[float],
    shrubs: Sequence[Sequence[float]],
    hour: float,
    width: int,
    height: int,
    field_of_view: float,
    segmentation_width: int | None = None,
    segmentation_height: int | None = None,
    frame_number: int = 0,
) -> ThermalFrame:
    """Render the first visible surface at every depth-camera pixel."""

    if len(depths) != width * height:
        raise ValueError("Depth image size does not match thermal image")
    if segmentation is not None:
        if segmentation_width is None or segmentation_height is None:
            raise ValueError("Segmentation dimensions are required")
        if len(segmentation) != segmentation_width * segmentation_height * 4:
            raise ValueError("Segmentation image size does not match its dimensions")

    if np is not None:
        return _render_temperature_frame_numpy(
            depths, segmentation, camera_position, camera_orientation, shrubs,
            hour, width, height, field_of_view, segmentation_width,
            segmentation_height, frame_number,
        )

    ambient_c = background_temperature_c(hour)
    sun = sunlight_fraction(hour)
    ambient_radiance = (ambient_c + 273.15) ** 4
    temperatures = [0.0] * (width * height)
    surfaces = bytearray(width * height)
    focal = width / (2.0 * math.tan(field_of_view / 2.0))
    orientation = camera_orientation
    origin = camera_position
    local_ys = [-(x + 0.5 - width / 2.0) / focal for x in range(width)]
    local_zs = [-(y + 0.5 - height / 2.0) / focal for y in range(height)]
    segmentation_xs = (
        [x * segmentation_width // width for x in range(width)]
        if segmentation is not None else None
    )
    shrub_regions = []
    for center in shrubs:
        delta = [center[axis] - origin[axis] for axis in range(3)]
        forward = sum(orientation[axis * 3] * delta[axis] for axis in range(3))
        if forward + SHRUB_RADIUS_XY <= 0.0:
            continue
        if forward <= SHRUB_RADIUS_XY:
            shrub_regions.append((center, 0, width - 1, 0, height - 1))
            continue
        right = -sum(orientation[axis * 3 + 1] * delta[axis] for axis in range(3))
        up = sum(orientation[axis * 3 + 2] * delta[axis] for axis in range(3))
        center_x = width / 2.0 + focal * right / forward
        center_y = height / 2.0 - focal * up / forward
        # The generous radius includes perspective growth at the near edge.
        radius = 2.0 * focal * SHRUB_RADIUS_XY / (forward - SHRUB_RADIUS_XY)
        x0 = max(0, math.floor(center_x - radius))
        x1 = min(width - 1, math.ceil(center_x + radius))
        y0 = max(0, math.floor(center_y - radius))
        y1 = min(height - 1, math.ceil(center_y + radius))
        if x0 <= x1 and y0 <= y1:
            shrub_regions.append((center, x0, x1, y0, y1))

    for y in range(height):
        local_z = local_zs[y]
        segmentation_y = (y * segmentation_height // height) if segmentation is not None else 0
        for x in range(width):
            index = y * width + x
            depth = depths[index]
            if not math.isfinite(depth) or depth <= 0.0:
                surface = SKY
                point = origin
                distance = 500.0
            else:
                local_y = local_ys[x]
                direction = (
                    orientation[0] + orientation[1] * local_y + orientation[2] * local_z,
                    orientation[3] + orientation[4] * local_y + orientation[5] * local_z,
                    orientation[6] + orientation[7] * local_y + orientation[8] * local_z,
                )
                point = tuple(origin[axis] + depth * direction[axis] for axis in range(3))
                distance = depth * math.sqrt(1.0 + local_y * local_y + local_z * local_z)
                surface = TERRAIN
                if segmentation is not None:
                    pixel = (segmentation_y * segmentation_width + segmentation_xs[x]) * 4
                    if segmentation[pixel] | segmentation[pixel + 1] | segmentation[pixel + 2]:
                        surface = DEER
                if surface != DEER:
                    for shrub_center, x0, x1, y0, y1 in shrub_regions:
                        if x < x0 or x > x1 or y < y0 or y > y1:
                            continue
                        entry = ray_ellipsoid_depth(origin, direction, shrub_center,
                                                    SHRUB_RADIUS_XY, SHRUB_RADIUS_Z)
                        # The depth camera decides visibility; the canopy geometry
                        # only labels a depth that is already on a visible shrub.
                        if entry is not None and abs(entry - depth) <= max(1.2, 0.015 * depth):
                            surface = SHRUB
                            break

            temperature = surface_temperature_c(surface, point, sun, ambient_c)
            # Reflected and atmospheric temperatures both equal the ambient
            # reference here, so their radiance terms can be combined.
            transmission = math.exp(-0.00018 * distance)
            radiance = ambient_radiance + transmission * EMISSIVITY[surface] * (
                (temperature + 273.15) ** 4 - ambient_radiance
            )
            apparent = radiance ** 0.25 - 273.15
            # A small, deterministic detector variation makes consecutive frames
            # distinct without tying apparent temperature to RGB brightness.
            noise = (((x * 73 + y * 151 + frame_number * 37) % 101) / 100.0 - 0.5) * 0.24
            temperatures[index] = apparent + noise
            surfaces[index] = surface

    return ThermalFrame(width, height, temperatures, surfaces, ambient_c)


def _render_temperature_frame_numpy(
    depths, segmentation, origin, orientation, shrubs, hour, width, height,
    field_of_view, segmentation_width, segmentation_height, frame_number,
) -> ThermalFrame:
    """Calculate the same surface model in arrays when NumPy is available."""

    ambient_c = background_temperature_c(hour)
    sun = sunlight_fraction(hour)
    depth = np.asarray(depths, dtype=np.float64).reshape(height, width)
    valid = np.isfinite(depth) & (depth > 0.0)
    safe_depth = np.where(valid, depth, 0.0)
    focal = width / (2.0 * math.tan(field_of_view / 2.0))
    local_y = (-(np.arange(width) + 0.5 - width / 2.0) / focal)[None, :]
    local_z = (-(np.arange(height) + 0.5 - height / 2.0) / focal)[:, None]
    directions = tuple(
        orientation[row * 3] + orientation[row * 3 + 1] * local_y
        + orientation[row * 3 + 2] * local_z
        for row in range(3)
    )
    points = tuple(origin[axis] + safe_depth * directions[axis] for axis in range(3))
    surfaces = np.full((height, width), TERRAIN, dtype=np.uint8)
    surfaces[~valid] = SKY

    if segmentation is not None:
        mask = np.frombuffer(segmentation, dtype=np.uint8).reshape(
            segmentation_height, segmentation_width, 4
        )
        sample_y = np.arange(height) * segmentation_height // height
        sample_x = np.arange(width) * segmentation_width // width
        visible_deer = np.any(mask[sample_y[:, None], sample_x[None, :], :3], axis=2)
        surfaces[valid & visible_deer] = DEER

    for center in shrubs:
        # Intersect camera rays with the shrub canopy, then check that the
        # depth camera actually sees that surface rather than one behind it.
        offset = ((origin[0] - center[0]) / SHRUB_RADIUS_XY,
                  (origin[1] - center[1]) / SHRUB_RADIUS_XY,
                  (origin[2] - center[2]) / SHRUB_RADIUS_Z)
        ray = (directions[0] / SHRUB_RADIUS_XY,
               directions[1] / SHRUB_RADIUS_XY,
               directions[2] / SHRUB_RADIUS_Z)
        a = ray[0] ** 2 + ray[1] ** 2 + ray[2] ** 2
        b = 2.0 * (offset[0] * ray[0] + offset[1] * ray[1] + offset[2] * ray[2])
        c = offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2 - 1.0
        discriminant = b ** 2 - 4.0 * a * c
        root = np.sqrt(np.maximum(discriminant, 0.0))
        near = (-b - root) / (2.0 * a)
        far = (-b + root) / (2.0 * a)
        visible_canopy = (
            valid & (surfaces != DEER) & (discriminant >= 0.0) & (far > 0.0)
            & (np.abs(np.maximum(near, 0.0) - safe_depth)
               <= np.maximum(1.2, 0.015 * safe_depth))
        )
        surfaces[visible_canopy] = SHRUB

    temperatures = np.empty((height, width), dtype=np.float64)
    sky = surfaces == SKY
    deer = surfaces == DEER
    shrub = surfaces == SHRUB
    terrain = surfaces == TERRAIN
    temperatures[sky] = ambient_c - 16.0
    temperatures[deer] = (
        27.5 + 0.12 * (ambient_c - 19.0) + 0.6 * sun
        + 0.35 * np.sin(points[0][deer] * 3.1 + points[1][deer] * 2.3
                        + points[2][deer])
    )
    temperatures[shrub] = (
        ambient_c - 1.2 + 2.0 * sun
        + 0.5 * np.sin(points[0][shrub] * 0.7 + points[1][shrub] * 0.4)
    )
    temperatures[terrain] = (
        ambient_c - 0.8 + 5.5 * sun
        + 0.9 * np.sin(points[0][terrain] * 0.045)
        * np.cos(points[1][terrain] * 0.055)
        - 0.006 * (points[2][terrain] - 200.0)
    )

    distance = np.where(valid, safe_depth * np.sqrt(1.0 + local_y ** 2 + local_z ** 2), 500.0)
    ambient_radiance = (ambient_c + 273.15) ** 4
    radiance = ambient_radiance + np.exp(-0.00018 * distance) * np.take(
        EMISSIVITY, surfaces
    ) * ((temperatures + 273.15) ** 4 - ambient_radiance)
    image_y = np.arange(height)[:, None]
    image_x = np.arange(width)[None, :]
    noise = (((image_x * 73 + image_y * 151 + frame_number * 37) % 101)
             / 100.0 - 0.5) * 0.24
    apparent = radiance ** 0.25 - 273.15 + noise
    return ThermalFrame(width, height, apparent.ravel().tolist(),
                        bytearray(surfaces.tobytes()), ambient_c)
