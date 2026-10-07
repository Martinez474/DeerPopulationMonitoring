#!/usr/bin/env python3

from osgeo import gdal
import math

INPUT_FILE = "catalina-512.tif"
OUTPUT_FILE = "CatalinaTerrain.proto"

# Change this if you want the island smaller in Webots.
HORIZONTAL_SCALE = 1.0

# Leave at 1.0 for realistic elevations.
# Increase it slightly if the terrain looks too flat.
VERTICAL_SCALE = 1.0

dataset = gdal.Open(INPUT_FILE)

if dataset is None:
    raise RuntimeError(f"Could not open {INPUT_FILE}")

band = dataset.GetRasterBand(1)
heights = band.ReadAsArray().astype(float)

width = dataset.RasterXSize
height = dataset.RasterYSize
transform = dataset.GetGeoTransform()
nodata = band.GetNoDataValue()

longitude_spacing = abs(transform[1])
latitude_spacing = abs(transform[5])

center_latitude = transform[3] + transform[5] * height / 2

meters_per_longitude_degree = 111320.0 * math.cos(math.radians(center_latitude))
meters_per_latitude_degree = 110574.0

x_spacing = longitude_spacing * meters_per_longitude_degree * HORIZONTAL_SCALE

y_spacing = latitude_spacing * meters_per_latitude_degree * HORIZONTAL_SCALE

# Make sea level zero and eliminate invalid pixels.
for row in range(height):
    for column in range(width):
        value = heights[row, column]

        if (
            not math.isfinite(value)
            or (nodata is not None and value == nodata)
            or value < 0
        ):
            value = 0.0

        heights[row, column] = value * VERTICAL_SCALE

with open(OUTPUT_FILE, "w", encoding="utf-8") as output:
    output.write("#VRML_SIM R2025a utf8\n\n")

    output.write(
        """PROTO CatalinaTerrain [
  field SFVec3f translation 0 0 0
] {
  Solid {
    translation IS translation
    name "Catalina terrain"
    children [
      Shape {
        appearance PBRAppearance {
          baseColor 0.30 0.48 0.20
          roughness 1
          metalness 0
        }
        geometry ElevationGrid {
"""
    )

    output.write(f"          xDimension {width}\n")
    output.write(f"          yDimension {height}\n")
    output.write(f"          xSpacing {x_spacing:.6f}\n")
    output.write(f"          ySpacing {y_spacing:.6f}\n")
    output.write("          height [\n")

    # Flip vertically so north/south orientation is preserved.
    flipped = heights[::-1, :]

    for row in flipped:
        values = " ".join(f"{value:.3f}" for value in row)
        output.write(f"            {values}\n")

    output.write(
        """          ]
          thickness 5
        }
      }
    ]
    boundingObject ElevationGrid {
"""
    )

    output.write(f"      xDimension {width}\n")
    output.write(f"      yDimension {height}\n")
    output.write(f"      xSpacing {x_spacing:.6f}\n")
    output.write(f"      ySpacing {y_spacing:.6f}\n")
    output.write("      height [\n")

    for row in flipped:
        values = " ".join(f"{value:.3f}" for value in row)
        output.write(f"        {values}\n")

    output.write(
        """      ]
      thickness 5
    }
  }
}
"""
    )

print(f"Created {OUTPUT_FILE}")
print(f"Grid: {width} × {height}")
print(f"Spacing: {x_spacing:.2f} m × {y_spacing:.2f} m")
print(
    f"Terrain size: {x_spacing * (width - 1):.2f} m × {y_spacing * (height - 1):.2f} m"
)
