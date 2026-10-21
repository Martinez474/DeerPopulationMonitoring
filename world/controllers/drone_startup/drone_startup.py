"""Select one controller for the world's single drone, then reload the world."""

from pathlib import Path
import sys

DRONE_MODE = "patrol"  # "patrol" or "drone_tracker"


def main():
    controllers = Path(__file__).resolve().parents[1]
    if DRONE_MODE == "patrol":
        sys.path.insert(0, str(controllers / "maveric_patrol"))
        from maveric_patrol import Mavic

        drone = Mavic()
    elif DRONE_MODE == "drone_tracker":
        sys.path.insert(0, str(controllers / "drone_tracker"))
        from drone_tracker import WildlifeSurveyDrone

        # The manual survey starts near the herd, as in the detection branch.
        drone = WildlifeSurveyDrone(survey_start=True)
    else:
        raise ValueError(f"Unknown DRONE_MODE {DRONE_MODE!r}; use 'patrol' or 'drone_tracker'")
    print(f"[startup] Drone mode: {DRONE_MODE}")
    drone.run()


if __name__ == "__main__":
    main()
