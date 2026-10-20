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
owns its duration, transitions, and movement intent.\