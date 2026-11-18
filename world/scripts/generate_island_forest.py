"""Generate terrain-height tree positions for Webots' Forest node.

Run from any directory: python3 scripts/generate_island_forest.py
Forest.proto negates the Y coordinate from treesFiles, so write negative Y.
"""

from pathlib import Path
import random
import re


ROOT = Path(__file__).resolve().parents[1]
TREE_COUNT = 100_000
SEED = 42


def main():
    terrain = (ROOT / "protos/CatalinaTerrain.proto").read_text()

    def field(name, convert):
        return convert(re.search(rf"\b{name}\s+([\d.]+)", terrain)[1])

    nx, ny = field("xDimension", int), field("yDimension", int)
    dx, dy = field("xSpacing", float), field("ySpacing", float)
    heights = list(map(float, re.search(r"height\s*\[([^]]+)\]", terrain)[1].split()))
    assert len(heights) == nx * ny
    land = [(i % nx, i // nx) for i, height in enumerate(heights) if height > 0]
    xmin, xmax = min(x for x, y in land) * dx, max(x for x, y in land) * dx
    ymin, ymax = min(y for x, y in land) * dy, max(y for x, y in land) * dy

    def elevation(x, y):
        gx, gy = x / dx, y / dy
        ix, iy = min(int(gx), nx - 2), min(int(gy), ny - 2)
        u, v = gx - ix, gy - iy
        a, b = heights[iy * nx + ix:iy * nx + ix + 2]
        c, d = heights[(iy + 1) * nx + ix:(iy + 1) * nx + ix + 2]
        return (1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * c + u * d)

    rng = random.Random(SEED)
    destination = ROOT / "worlds/forest/catalina.forest"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as output:
        for _ in range(TREE_COUNT):
            x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
            # Bilinear terrain sampling; no shoreline filtering.
            output.write(f"{x:.3f},{-y:.3f},{elevation(x, y):.3f}\n")
    print(f"Generated {TREE_COUNT:,} trees across {xmax-xmin:.1f} x {ymax-ymin:.1f} m")


if __name__ == "__main__":
    main()
