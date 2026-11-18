# Deer Population Monitoring

Survey deer on Catalina Island with a manually flown drone in Webots. The
prototype creates a simulated thermal image and marks possible animals.

> **Status:** Temperatures are illustrative. Thermal results are heat candidates;
> YOLO26 and animal tracking are planned.

## Get started

Use **Webots with Python controllers**. NumPy is optional and accelerates
rendering when installed in the Python environment Webots uses.

1. Open [CatalinaIsland.wbt](world/worlds/CatalinaIsland.wbt).
2. Click **Reload World**, then **Play** if paused.
3. Thermal starts automatically. Look for **Drone thermal view**.
4. Click the main 3D view before using the keyboard.

## Controls

| Key | Action |
| --- | --- |
| Up / Down | Fly forward / backward |
| Left / Right | Turn left / right |
| Shift + Left / Right | Move sideways |
| Shift + Up / Down | Raise / lower the drone |
| **T** | Switch thermal / RGB |
| **O** | Advance the clock six hours |

## How it works

**See → assign temperature → find warm regions → show candidates**

1. **See:** Depth identifies the first visible surface. Webots segmentation
   marks visible deer pixels; shrub geometry identifies canopy surfaces.
2. **Assign temperature:** Formulas simulate ground, shrub, fur, and sky
   temperatures, including daytime warming and approximate infrared effects.
3. **Find warm regions:** After a slight blur, the detector groups unusually hot
   pixels and filters by size and shape. It uses temperatures and depth, without
   deer labels or known positions.
4. **Show candidates:** Orange `+` marks and the count describe possible animals
   in the current frame. RGB mode separately uses Webots recognition labels.

The purple-to-yellow color scale adjusts with the background temperature.
Check its range on the HUD when comparing frames. A simulated day starts at
**06:00** and lasts **ten simulation minutes**.

## Current limits

- Warm ground can cause false alarms. Hidden or distant deer can be missed.
- Counts are per frame; the system does not yet identify unique animals.
- The visible sun shares the thermal clock. Heating follows preset rules,
  without stored heat or gradual cooling. Sky artwork stays fixed; shadows are off.
- Known deer positions only score a console diagnostic and never alter detections.

## Performance

Thermal depth and display use **256 × 144 pixels**. RGB uses **640 × 360**, with
antialiasing off. Sensors update every **256 ms** with this world's 8 ms step.
NumPy accelerates thermal rendering; a Python fallback works without it.

## Where to look

| File | Purpose |
| --- | --- |
| [drone_tracker.py](world/controllers/drone_tracker/drone_tracker.py) | Flight, sensors, clock, and HUD |
| [thermal_scene.py](world/controllers/drone_tracker/thermal_scene.py) | Temperature image |
| [thermal_image_detection.py](world/controllers/drone_tracker/thermal_image_detection.py) | Hot-region detection |
| [thermal_view.py](world/controllers/drone_tracker/thermal_view.py) | Blur and display colors |

## Next steps

1. Add real solar position, weather, material properties, shade, and gradual heating.
2. Validate against independent thermal observations.
3. Validate or train YOLO26 for thermal images, then consider Kalman tracking.

## Quick fixes

- **Keyboard unresponsive:** Click the main 3D view.
- **Sensor overlay blocks the scene:** Close the overlay with **X**.
- **Need thermal view:** Press **T**. **Plain Rendering** only changes drawing style.
