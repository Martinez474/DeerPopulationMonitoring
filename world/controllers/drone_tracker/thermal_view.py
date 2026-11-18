"""Simulated thermal optics and the false-color display palette."""

from typing import Sequence


PALETTE = (
    (0.00, (9, 15, 58)),
    (0.22, (54, 21, 106)),
    (0.44, (137, 30, 132)),
    (0.62, (232, 65, 69)),
    (0.80, (255, 159, 31)),
    (0.93, (255, 228, 69)),
    (1.00, (255, 252, 225)),
)
PALETTE_RGB = []
for value in range(256):
    position = value / 255.0
    for (start, low), (end, high) in zip(PALETTE, PALETTE[1:]):
        if position <= end:
            amount = (position - start) / (end - start)
            PALETTE_RGB.append(
                tuple(round(low[channel] + amount * (high[channel] - low[channel])) for channel in range(3))
            )
            break
PALETTE_BYTES = tuple(bytes(color) for color in PALETTE_RGB)


def display_temperature_range(ambient_c: float) -> tuple[float, float]:
    """Keep useful contrast as the simulated day warms or cools."""

    return ambient_c - 8.0, ambient_c + 16.0


def temperature_index(temperature_c: float, low_c: float = 4.0,
                      high_c: float = 40.0) -> int:
    """Map apparent Celsius to one color in the display palette."""

    return max(0, min(255, round((temperature_c - low_c) * 255.0 / (high_c - low_c))))


def blur_temperatures(temperatures: Sequence[float], width: int, height: int) -> list[float]:
    """Apply a small separable point-spread blur before display and detection."""

    if len(temperatures) != width * height:
        raise ValueError("Temperature image size does not match dimensions")
    horizontal = [0.0] * len(temperatures)
    result = [0.0] * len(temperatures)
    for y in range(height):
        row = y * width
        for x in range(width):
            left = row + max(0, x - 1)
            right = row + min(width - 1, x + 1)
            horizontal[row + x] = (temperatures[left] + 3.0 * temperatures[row + x]
                                   + temperatures[right]) / 5.0
    for y in range(height):
        above = max(0, y - 1) * width
        row = y * width
        below = min(height - 1, y + 1) * width
        for x in range(width):
            result[row + x] = (horizontal[above + x] + 3.0 * horizontal[row + x]
                               + horizontal[below + x]) / 5.0
    return result


def colorize_temperatures(temperatures: Sequence[float], width: int, height: int,
                          low_c: float = 4.0, high_c: float = 40.0) -> bytes:
    """Turn apparent temperatures into RGB pixels for the Webots Display."""

    if len(temperatures) != width * height:
        raise ValueError("Temperature image size does not match dimensions")
    if high_c <= low_c:
        raise ValueError("Display temperature range must increase")
    rgb = bytearray(width * height * 3)
    scale = 255.0 / (high_c - low_c)
    for y in range(height):
        row = y * width
        legend = PALETTE_BYTES[255 - y * 255 // max(1, height - 1)]
        for x in range(width):
            pixel = (row + x) * 3
            if x == width - 11:
                rgb[pixel:pixel + 3] = b"\x05\x05\x11"
            elif x >= width - 10:
                rgb[pixel:pixel + 3] = legend
            else:
                palette_index = max(0, min(255, round((temperatures[row + x] - low_c) * scale)))
                rgb[pixel:pixel + 3] = PALETTE_BYTES[palette_index]
    return bytes(rgb)
