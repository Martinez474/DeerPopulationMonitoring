"""Start the single manually controlled wildlife survey drone."""

from pathlib import Path
import sys

def main():
    controllers = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(controllers / "drone_tracker"))
    from drone_tracker import WildlifeSurveyDrone

    # Place the manually flown drone near the herd.
    drone = WildlifeSurveyDrone(survey_start=True)
    print("[startup] Manual drone survey ready")
    drone.run()


if __name__ == "__main__":
    main()
