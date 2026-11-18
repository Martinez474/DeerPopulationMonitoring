"""Manual flight, deer detection, and HUD for the survey drone."""

from dataclasses import dataclass
import math

from controller import Keyboard, Supervisor

from detection import THERMAL_MODE, VisualShapeDeerDetector
from thermal_detection import (
    DEER_DEFS,
    background_temperature_c,
    count_ground_truth_matches,
    hour_of_day,
    sunlight_direction,
    sunlight_fraction,
)
from thermal_image_detection import ThermalImageDeerDetector
from thermal_scene import render_temperature_frame
from thermal_view import blur_temperatures, colorize_temperatures, display_temperature_range
from view_control import DroneViewController


DETECTION_BACKEND = "yolo"  # RGB backend: "yolo" or "visual_shape"
YOLO_INTERVAL_SECONDS = 0.2

HUD_FONT = "Arial Black"
HUD_TEXT = 0xFFFFFF
HUD_ACCENT = 0x9EEBFF
HUD_HIGHLIGHT = 0xFFF176
HUD_HEAT = 0xFF6B38

# change this value to resize all hud text
HUD_TEXT_SCALE = 1.5


def clamp(value: float, low: float, high: float) -> float:
    """Keep a number inside a fixed range."""

    return max(low, min(high, value))


@dataclass
class FlightControls:
    """Movement requested during one simulation step."""

    forward: float = 0.0
    strafe: float = 0.0
    climb: float = 0.0
    turn: float = 0.0


class WildlifeSurveyDrone(Supervisor):
    """Control the drone and coordinate its camera tools."""

    # movement speeds
    forward_speed = 8.0
    strafe_speed = 6.0
    climb_speed = 3.0
    turn_speed = 1.1

    # altitude limits
    min_altitude = 204.0
    max_altitude = 500.0

    # slower camera updates reduce rendering work
    camera_update_steps = 32
    hud_update_interval = 1.0
    action_cooldown = 0.25

    def __init__(self, survey_start=False):
        super().__init__()

        if survey_start:
            # Reuse the same drone, placing it at the survey start.
            drone = self.getSelf()
            drone.getField("translation").setSFVec3f([23868, 17007, 220])
            drone.getField("rotation").setSFRotation([0, 0, 1, 0])
            drone.resetPhysics()

        # measure one simulation step
        self.time_step = int(self.getBasicTimeStep())
        self.dt = self.time_step / 1000.0

        # prepare drone devices
        self._setup_drone_pose()
        self._setup_camera()
        self._start_propellers()

        self.leds = (
            self.getDevice("front left led"),
            self.getDevice("front right led"),
        )
        self.last_led_state = None

        # connect keyboard input
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.time_step)

        # track timed updates
        self.next_action_time = 0.0
        self.next_hud_update = 0.0
        self.last_deer_count = None
        self.last_hud_mode = None
        self.thermal_time_offset = 0.0
        self.thermal_display_range = None
        self._setup_sun()

        self._clear_old_hud_labels()
        self.print_controls()

    def _setup_drone_pose(self) -> None:
        """Store the fields used for stable, kinematic flight."""

        # keep access to the drone pose fields
        self.drone = self.getSelf()
        self.translation = self.drone.getField("translation")
        self.rotation = self.drone.getField("rotation")
        self.position = list(self.translation.getSFVec3f())

        # preserve the starting yaw direction
        starting_rotation = self.rotation.getSFRotation()
        direction = 1.0 if starting_rotation[2] >= 0 else -1.0
        self.yaw = starting_rotation[3] * direction

    def _setup_camera(self) -> None:
        """Start the camera, recognition, gimbal, and view controller."""

        # update camera and recognition together
        self.camera = self.getDevice("camera")
        camera_period = self.time_step * self.camera_update_steps
        self.camera.enable(camera_period)
        self.camera.recognitionEnable(camera_period)
        self.camera.enableRecognitionSegmentation()
        self.range_finder = self.getDevice("thermal depth")
        self.range_finder.enable(camera_period)

        # Thermal uses temperature pixels; RGB uses the configured detector.
        # Load YOLO lazily when RGB is first selected.
        self.detectors = {
            "visual_shape": VisualShapeDeerDetector(),
            THERMAL_MODE: ThermalImageDeerDetector(),
        }
        self.detector = self.detectors[THERMAL_MODE]
        self.detections = []
        self.detection_interval = camera_period / 1000.0
        self.next_detection_time = camera_period / 1000.0
        self.thermal_frame_number = 0
        self.shrubs = [
            node for number in range(1, 4)
            if (node := self.getFromDef(f"THERMAL_SHRUB_{number}")) is not None
        ]
        self.ground_truth_deer_nodes = [
            node for name in DEER_DEFS if (node := self.getFromDef(name)) is not None
        ]

        # The 3D viewpoint always follows the onboard camera.
        self.view = DroneViewController(self, self.camera, "SURVEY_VIEWPOINT")

        # Display pixels texture a panel behind the sensing camera. The 3D
        # viewpoint looks at that panel whenever thermal mode is selected.
        self.thermal_display = self.getDevice("thermal display")
        display_node = self.getFromDevice(self.thermal_display._tag)
        if display_node is None:
            raise RuntimeError("Could not access the thermal display node")
        self.thermal_screen_position = display_node.getField("translation")
        self.thermal_screen_visible = False
        self.next_thermal_frame_time = 0.0
        self.thermal_frame_interval = self.detection_interval
        self._sync_thermal_screen()

        # Keep the survey camera aimed at the terrain.
        self.getDevice("camera roll").setPosition(0.0)
        self.getDevice("camera pitch").setPosition(1.15)

    def _start_propellers(self) -> None:
        """Spin the propellers so the drone looks active."""

        # opposite directions match the propeller layout
        propellers = (
            ("front left propeller", 35.0),
            ("front right propeller", -35.0),
            ("rear left propeller", -35.0),
            ("rear right propeller", 35.0),
        )
        for name, speed in propellers:
            motor = self.getDevice(name)
            motor.setPosition(float("inf"))
            motor.setVelocity(speed)

    def _setup_sun(self) -> None:
        """Connect the visible sun to the thermal survey clock."""

        sun = self.getFromDef("SURVEY_SUN")
        if sun is None:
            raise RuntimeError("Could not find the survey sun")
        self.sun_direction = sun.getField("direction")
        self.sun_intensity = sun.getField("intensity")
        self.sun_color = sun.getField("color")
        self.sun_on = sun.getField("on")
        self.next_sun_update = 0.0
        self.update_sun(self.getTime())

    def update_sun(self, now: float) -> None:
        """Move and brighten the visible sun as solar heating changes."""

        if now < self.next_sun_update:
            return
        hour = hour_of_day(now, self.thermal_time_offset)
        strength = sunlight_fraction(hour)
        self.sun_direction.setSFVec3f(sunlight_direction(hour))
        self.sun_intensity.setSFFloat(1.5 * strength)
        self.sun_color.setSFColor([1.0, 0.75 + 0.25 * strength,
                                   0.55 + 0.45 * strength])
        self.sun_on.setSFBool(strength > 0.01)
        self.next_sun_update = now + 1.0

    def _clear_old_hud_labels(self) -> None:
        """Remove labels left behind if this controller is restarted."""

        for label_id in range(32):
            self.setLabel(label_id, "", 0.0, 0.0, 0.01, HUD_TEXT)

    @staticmethod
    def print_controls() -> None:
        """Show the keyboard controls in the Webots console."""

        print("\nWildlife survey drone ready in thermal view.")
        print("Close any yellow depth or magenta RGB overlay with its X, then click the 3D view.")
        print("Use these keys to control the drone:")
        print("  Up / Down                 fly forward / backward")
        print("  Left / Right              turn left / right")
        print("  Shift + Left / Right      strafe left / right")
        print("  Shift + Up / Down         raise / lower the drone")
        print("  T                         switch thermal / RGB view")
        print("  O                         advance survey clock by six hours")
        print("  No activation key is needed: thermal is the starting view.\n")

    def read_controls(self, now: float) -> FlightControls:
        """Read every queued key and return the requested movement."""

        # collect movement from every key in this step
        controls = FlightControls()

        key = self.keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                controls.forward = 1.0
            elif key == Keyboard.DOWN:
                controls.forward = -1.0
            elif key == Keyboard.LEFT:
                controls.turn = 1.0
            elif key == Keyboard.RIGHT:
                controls.turn = -1.0
            elif key == (Keyboard.SHIFT | Keyboard.LEFT):
                controls.strafe = -1.0
            elif key == (Keyboard.SHIFT | Keyboard.RIGHT):
                controls.strafe = 1.0
            elif key == (Keyboard.SHIFT | Keyboard.UP):
                controls.climb = 1.0
            elif key == (Keyboard.SHIFT | Keyboard.DOWN):
                controls.climb = -1.0
            elif key in (ord("T"), ord("t")):
                self._toggle_detector(now)
            elif key in (ord("O"), ord("o")):
                self._advance_clock(now)

            key = self.keyboard.getKey()

        return controls

    def _toggle_detector(self, now: float) -> None:
        if now < self.next_action_time:
            return
        self.next_action_time = now + self.action_cooldown
        next_mode = DETECTION_BACKEND if self.detector.mode == THERMAL_MODE else THERMAL_MODE
        if next_mode not in self.detectors:
            if next_mode != "yolo":
                raise ValueError(f"Unknown RGB detection backend: {next_mode}")
            from yolo_detection import YoloDeerDetector
            self.detectors[next_mode] = YoloDeerDetector()
        self.detector = self.detectors[next_mode]
        self.detection_interval = self.time_step * self.camera_update_steps / 1000.0
        if next_mode == "yolo":
            self.detection_interval = max(self.detection_interval, YOLO_INTERVAL_SECONDS)
        self.view.thermal_mode = next_mode == THERMAL_MODE
        self._sync_thermal_screen()
        self.view.update()
        self.next_detection_time = 0.0
        self.next_thermal_frame_time = 0.0
        self.next_hud_update = 0.0
        self.detections = []
        print(f"[detection] {next_mode}")

    def _advance_clock(self, now: float) -> None:
        if now < self.next_action_time:
            return
        self.next_action_time = now + self.action_cooldown
        self.thermal_time_offset += 6.0
        self.next_detection_time = 0.0
        self.next_thermal_frame_time = 0.0
        self.next_hud_update = 0.0
        self.next_sun_update = 0.0

    def _sync_thermal_screen(self) -> None:
        visible = self.detector.mode == THERMAL_MODE
        if visible == self.thermal_screen_visible:
            return
        self.thermal_screen_position.setSFVec3f(
            [-0.5, 0.0, 0.0] if visible else [0.0, 0.0, 1000.0]
        )
        self.thermal_screen_visible = visible

    def update_thermal_view(self, now: float) -> None:
        if not self.thermal_screen_visible or now < self.next_thermal_frame_time:
            return
        depths = self.range_finder.getRangeImage()
        if not depths:
            return
        segmentation = self.camera.getRecognitionSegmentationImage()
        display = self.thermal_display
        width, height = self.range_finder.getWidth(), self.range_finder.getHeight()
        hour = hour_of_day(now, self.thermal_time_offset)
        frame = render_temperature_frame(
            depths, segmentation,
            self.view.camera_node.getPosition(),
            self.view.camera_node.getOrientation(),
            [tuple(node.getPosition()[axis] + (1.8 if axis == 2 else 0.0)
                   for axis in range(3)) for node in self.shrubs],
            hour, width, height, self.camera.getFov(),
            self.camera.getWidth(), self.camera.getHeight(),
            self.thermal_frame_number,
        )
        temperatures = blur_temperatures(frame.temperatures, width, height)
        self.thermal_display_range = display_temperature_range(frame.background_c)
        pixels = colorize_temperatures(temperatures, width, height,
                                       *self.thermal_display_range)
        image_ref = display.imageNew(
            pixels, display.RGB, width, height
        )
        display.imagePaste(image_ref, 0, 0, False)
        display.imageDelete(image_ref)
        self.detections = self.detector.detect(
            temperatures, depths, width, height, self.camera.getFov(),
            self.camera.getWidth(), self.camera.getHeight(), frame.background_c,
        )
        self.thermal_frame_number += 1
        self.next_thermal_frame_time = now + self.thermal_frame_interval

    def move_drone(self, controls: FlightControls) -> None:
        """Apply the requested movement in the drone's local direction."""

        # turn before calculating movement
        self.yaw += controls.turn * self.turn_speed * self.dt
        self.yaw = (self.yaw + math.pi) % (2.0 * math.pi) - math.pi

        # convert local movement into world movement
        forward_distance = controls.forward * self.forward_speed * self.dt
        strafe_distance = controls.strafe * self.strafe_speed * self.dt
        heading_cos = math.cos(self.yaw)
        heading_sin = math.sin(self.yaw)

        self.position[0] += (
            heading_cos * forward_distance + heading_sin * strafe_distance
        )
        self.position[1] += (
            heading_sin * forward_distance - heading_cos * strafe_distance
        )
        # keep altitude inside safe limits
        self.position[2] = clamp(
            self.position[2] + controls.climb * self.climb_speed * self.dt,
            self.min_altitude,
            self.max_altitude,
        )

        # apply the pose directly for stable flight
        self.translation.setSFVec3f(self.position)
        self.rotation.setSFRotation([0.0, 0.0, 1.0, self.yaw])
        self.drone.resetPhysics()

    def update_leds(self, now: float) -> None:
        """Alternate the front lights once per second."""

        led_state = int(now) % 2

        # skip device writes when the state is unchanged
        if led_state == self.last_led_state:
            return
        self.leds[0].set(led_state)
        self.leds[1].set(1 - led_state)
        self.last_led_state = led_state

    def update_detection(self, now: float) -> None:
        """Refresh detections at the camera sampling rate."""

        # only scan after a new camera frame
        if now < self.next_detection_time:
            return
        if self.detector.mode != THERMAL_MODE:
            self.detections = self.detector.detect(self.camera)
        self.next_detection_time = now + self.detection_interval

    def set_hud_text(
        self,
        label_id: int,
        text: str,
        x: float,
        y: float,
        size: float,
        color: int,
    ) -> None:
        """Draw one crisp HUD label without outlines or shadows."""

        # apply one scale to every label
        scaled_size = size * HUD_TEXT_SCALE
        self.setLabel(label_id, text, x, y, scaled_size, color, 0.0, HUD_FONT)

    def update_hud(self, now: float) -> None:
        """Update the flight information shown over the 3D view."""

        deer_count = len(self.detections)
        count_changed = deer_count != self.last_deer_count
        mode_changed = self.detector.mode != self.last_hud_mode

        # refresh early when the deer count changes
        if not count_changed and not mode_changed and now < self.next_hud_update:
            return

        thermal = self.detector.mode == THERMAL_MODE
        view_text = "Drone thermal view" if thermal else "Drone visual view"
        hour = hour_of_day(now, self.thermal_time_offset)

        # keep related labels spaced with the text scale
        title_y = 0.025
        count_y = title_y + (0.045 * HUD_TEXT_SCALE) + 0.025
        controls_second_y = 0.945
        controls_first_y = controls_second_y - (0.026 * HUD_TEXT_SCALE) - 0.012

        # draw view status and deer count
        self.set_hud_text(0, view_text, 0.025, title_y, 0.045, HUD_ACCENT)
        scale_text = ""
        if thermal and self.thermal_display_range is not None:
            low_c, high_c = self.thermal_display_range
            scale_text = f"SCALE {low_c:.0f}-{high_c:.0f} C"
        self.set_hud_text(6, scale_text, 0.75, title_y, 0.024, HUD_TEXT)
        self.set_hud_text(
            1,
            f"{'Thermal candidates' if thermal else 'Deer visible'} ({self.detector.mode}): {deer_count}",
            0.025,
            count_y,
            0.078,
            HUD_HIGHLIGHT,
        )
        # draw controls at the bottom of the screen
        self.set_hud_text(
            3,
            "Move: arrows    Strafe / altitude: Shift + arrows",
            0.025,
            controls_first_y,
            0.026,
            HUD_TEXT,
        )
        self.set_hud_text(
            4,
            "Thermal starts automatically    T: thermal / RGB    O: time +6h",
            0.025,
            controls_second_y,
            0.026,
            HUD_TEXT,
        )
        # Show the reticle in the onboard view.
        self.set_hud_text(
            5,
            "+",
            0.487,
            0.455,
            0.072,
            HUD_HIGHLIGHT,
        )
        self.set_hud_text(
            2,
            (
                f"SIMULATED THERMAL  {int(hour):02d}:{int((hour % 1) * 60):02d}  "
                f"background {background_temperature_c(hour):.1f} C"
                if thermal else "VISUAL CAMERA"
            ),
            0.025,
            count_y + (0.078 * HUD_TEXT_SCALE) + 0.025,
            0.031,
            HUD_HEAT if thermal else HUD_ACCENT,
        )
        for index in range(20):
            marker = ""
            x = y = 0.0
            if thermal and index < len(self.detections):
                detection = self.detections[index]
                marker = "+"
                x = clamp(detection.image_x / self.camera.getWidth(), 0.0, 0.96)
                y = clamp(detection.image_y / self.camera.getHeight(), 0.0, 0.96)
            self.set_hud_text(10 + index, marker, x, y, 0.045, HUD_HEAT)

        if thermal:
            matches = count_ground_truth_matches(
                self.detections,
                [node.getPosition() for node in self.ground_truth_deer_nodes],
                self.view.camera_node.getPosition(),
                self.view.camera_node.getOrientation(),
                self.camera.getWidth(), self.camera.getHeight(), self.camera.getFov(),
            )
            print(f"[thermal] candidates: {deer_count}, near known deer: {matches} | "
                  f"altitude: {self.position[2]:.1f} m")
        else:
            print(f"[visual] deer visible: {deer_count} | altitude: {self.position[2]:.1f} m")
        self.last_deer_count = deer_count
        self.last_hud_mode = self.detector.mode
        self.next_hud_update = now + self.hud_update_interval

    def run(self) -> None:
        """Run one controller update for every simulation step."""

        # keep flight responsive while slower updates run
        while self.step(self.time_step) != -1:
            now = self.getTime()
            controls = self.read_controls(now)
            self.update_sun(now)
            camera_before = self.view.camera_node.getPosition()
            drone_before = tuple(self.position)
            self.move_drone(controls)
            self.update_leds(now)
            camera_after = self.view.camera_node.getPosition()
            correction = tuple(
                self.position[axis] - drone_before[axis]
                - (camera_after[axis] - camera_before[axis])
                for axis in range(3)
            )
            self.view.update(correction)
            self.update_detection(now)
            self.update_thermal_view(now)
            self.update_hud(now)


if __name__ == "__main__":
    # start the controller when webots runs this file
    WildlifeSurveyDrone().run()
