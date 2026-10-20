"""save drone camera frames and matching flight information."""

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from detection import DeerDetection


class FlightRecorder:
    """save camera images and a small row of data for each image."""

    fieldnames = (
        "simulation_time_s",
        "frame_number",
        "image_file",
        "detection_mode",
        "deer_count",
        "drone_x_m",
        "drone_y_m",
        "drone_altitude_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "detections_json",
    )

    def __init__(self, project_root: Path, frame_interval: float = 0.5):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.recordings_root = project_root / "recordings"
        self.session_directory = self.recordings_root / f"flight-{timestamp}"
        self.session_directory.mkdir(parents=True, exist_ok=True)
        self.latest_status_file = self.recordings_root / "latest_detection.json"

        self.frame_interval = frame_interval
        self.next_frame_time = 0.0
        self.frame_number = 0
        self.enabled = True

        telemetry_path = self.session_directory / "telemetry.csv"
        self.telemetry_file = telemetry_path.open("w", newline="", encoding="utf-8")
        self.telemetry = csv.DictWriter(
            self.telemetry_file, fieldnames=self.fieldnames
        )
        self.telemetry.writeheader()
        self.telemetry_file.flush()

        session_info = {
            "created_at": datetime.now().astimezone().isoformat(),
            "detection_mode": "visual_shape",
            "frame_interval_seconds": frame_interval,
            "image_format": "jpeg",
        }
        (self.session_directory / "session.json").write_text(
            json.dumps(session_info, indent=2) + "\n", encoding="utf-8"
        )

    def toggle(self, simulation_time: float) -> bool:
        """pause or resume automatic recording."""

        self.enabled = not self.enabled
        if self.enabled:
            self.next_frame_time = simulation_time
        return self.enabled

    def capture(
        self,
        camera: Any,
        simulation_time: float,
        detection_mode: str,
        detections: list[DeerDetection],
        position: tuple[float, float, float],
        yaw: float,
        force: bool = False,
    ) -> Path | None:
        """save one image and its data when the recording timer is ready."""

        if not force and (not self.enabled or simulation_time < self.next_frame_time):
            return None

        self.frame_number += 1
        self.next_frame_time = simulation_time + self.frame_interval
        image_name = f"frame-{self.frame_number:06d}.jpg"
        image_path = self.session_directory / image_name
        if camera.saveImage(str(image_path), 90) != 0:
            print(f"[recording] could not save {image_path}")
            return None

        detection_data = [detection.to_dict() for detection in detections]
        self.telemetry.writerow(
            {
                "simulation_time_s": f"{simulation_time:.3f}",
                "frame_number": self.frame_number,
                "image_file": image_name,
                "detection_mode": detection_mode,
                "deer_count": len(detections),
                "drone_x_m": f"{position[0]:.4f}",
                "drone_y_m": f"{position[1]:.4f}",
                "drone_altitude_m": f"{position[2]:.4f}",
                "roll_rad": "0.000000",
                "pitch_rad": "0.000000",
                "yaw_rad": f"{yaw:.6f}",
                "detections_json": json.dumps(detection_data, separators=(",", ":")),
            }
        )
        self.telemetry_file.flush()

        latest = {
            "simulation_time_seconds": round(simulation_time, 3),
            "recording": self.enabled,
            "detection_mode": detection_mode,
            "deer_count": len(detections),
            "drone_position_meters": {
                "x": position[0],
                "y": position[1],
                "altitude": position[2],
            },
            "detections": detection_data,
            "latest_frame": str(image_path.relative_to(self.recordings_root.parent)),
            "session_directory": str(
                self.session_directory.relative_to(self.recordings_root.parent)
            ),
        }
        self.latest_status_file.write_text(
            json.dumps(latest, indent=2) + "\n", encoding="utf-8"
        )
        return image_path

    def close(self) -> None:
        """close the telemetry file when the simulation stops."""

        self.telemetry_file.close()
