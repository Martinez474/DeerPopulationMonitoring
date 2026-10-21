## Select the drone mode

Edit `world/controllers/drone_startup/drone_startup.py`:

```python
DRONE_MODE = "patrol"  # "patrol" or "drone_tracker"
```

Save and reload the world (reset the simulation) after changing modes.
There is one drone. `patrol` starts on the launch pad and follows the existing
waypoints. `drone_tracker` places that drone near the deer for manual flight
and camera recognition. Detection is currently active only in tracker mode.
The launch pad and waypoint markers remain in both modes.

The controls below apply to `drone_tracker` mode.

## Drone Controls

| Input | Action |
| --- | --- |
| Up / Down | Fly forward / backward |
| Left / Right | Turn left / right |
| Shift + Left / Right | Strafe left / right |
| Shift + Up / Down | Raise / lower the drone |
| I / K | Tilt the camera up / down |
| V | Toggle between chase view and the drone camera |


## Deer Detection Design

The current `visual_shape` backend uses the simulator camera's rendered object
recognition to count only deer whose silhouettes are inside the frame and not
occluded. Detection is separated from flight control in
`world/controllers/drone_tracker/detection.py`

The deer controller uses separate `IdleState`, `GrazeState`, `AlertState`, and
`WalkState` classes in `world/controllers/deer_ai/deer_states.py`. Each class
owns its duration, transitions, and movement intent.
