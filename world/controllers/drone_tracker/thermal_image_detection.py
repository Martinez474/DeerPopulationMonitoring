"""Find compact hot regions in the simulated sensor image.

This detector sees apparent temperatures and depth only.  It never receives
the scene's deer node positions, recognition labels, or segmentation mask.
"""

import math
from statistics import median
from typing import Sequence

from detection import DeerDetection, THERMAL_MODE


class ThermalImageDeerDetector:
    mode = THERMAL_MODE

    def detect(self, temperatures: Sequence[float], depths: Sequence[float],
               width: int, height: int, fov: float,
               camera_width: int, camera_height: int,
               ambient_c: float) -> list[DeerDetection]:
        if len(temperatures) != width * height or len(depths) != width * height:
            raise ValueError("Thermal and depth images must match their dimensions")

        hot = bytearray(width * height)
        reference_radius = min(9, max(2, min(width, height) // 6))
        absolute_floor = max(21.0, ambient_c + 3.0)
        for y in range(height):
            for x in range(width):
                index = y * width + x
                temperature = temperatures[index]
                if temperature < absolute_floor or not math.isfinite(depths[index]):
                    continue
                left = temperatures[y * width + max(0, x - reference_radius)]
                right = temperatures[y * width + min(width - 1, x + reference_radius)]
                above = temperatures[max(0, y - reference_radius) * width + x]
                below = temperatures[min(height - 1, y + reference_radius) * width + x]
                # For four samples, the median is the mean of the middle two.
                local_median = (left + right + above + below
                                - min(left, right, above, below)
                                - max(left, right, above, below)) * 0.5
                if temperature - local_median >= 2.8:
                    hot[index] = 1

        detections = []
        visited = bytearray(len(hot))
        focal = width / (2.0 * math.tan(fov / 2.0))
        for index in range(len(hot)):
            if not hot[index] or visited[index]:
                continue
            pending = [index]
            visited[index] = 1
            pixels = []
            while pending:
                current = pending.pop()
                pixels.append(current)
                cy, cx = divmod(current, width)
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        neighbor = ny * width + nx
                        if hot[neighbor] and not visited[neighbor]:
                            visited[neighbor] = 1
                            pending.append(neighbor)

            if len(pixels) < 3 or len(pixels) > width * height * 0.04:
                continue
            xs = [pixel % width for pixel in pixels]
            ys = [pixel // width for pixel in pixels]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            if x1 - x0 > width // 4 or y1 - y0 > height // 3:
                continue
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            candidate_depths = [depths[pixel] for pixel in pixels if math.isfinite(depths[pixel])]
            depth = median(candidate_depths)
            peak = max(temperatures[pixel] for pixel in pixels)
            contrast = max(0.0, peak - ambient_c)
            detections.append(DeerDetection(
                object_id=-(len(detections) + 1),
                model="thermal_candidate",
                image_x=round(center_x * camera_width / width),
                image_y=round(center_y * camera_height / height),
                image_width=max(1, round((x1 - x0 + 1) * camera_width / width)),
                image_height=max(1, round((y1 - y0 + 1) * camera_height / height)),
                camera_x=depth,
                camera_y=-(center_x - width / 2.0) * depth / focal,
                camera_z=-(center_y - height / 2.0) * depth / focal,
                confidence=min(1.0, contrast / 15.0),
                thermal_contrast_c=round(contrast, 2),
            ))
        return detections
