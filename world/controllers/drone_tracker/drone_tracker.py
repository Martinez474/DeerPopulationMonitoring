"""Manual flight, deer detection, and HUD for the survey drone."""

from dataclasses import dataclass
import math

from controller import Keyboard, Supervisor

from detection import VisualShapeDeerDetector
from view_control import DRONE_VIEW, DroneViewController


HUD_FONT = "Arial Black"
HUD_TEXT = 0xFFFFFF
HUD_ACCENT = 0x9EEBFF
HUD_HIGHLIGHT = 0xFFF176

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

    # altitude and camera limits
    min_altitude = 204.0
    max_altitude = 500.0
    min_camera_pitch = -0.5
    max_camera_pitch = 1.7

    # slower camera updates reduce rendering work
    camera_update_steps = 8
    hud_update_interval = 1.0
    action_cooldown = 0.25

    def __init__(self):
        super().__init__()

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

        # keep deer detection separate from flight
        self.detector = VisualShapeDeerDetector()
        self.detections = []
        self.detection_interval = camera_period / 1000.0
        self.next_detection_time = 0.0

        # manage chase and onboard views
        self.view = DroneViewController(self, self.camera, "SURVEY_VIEWPOINT")

        # point the camera toward the terrain
        self.camera_pitch = 1.15
        self.getDevice("camera roll").setPosition(0.0)
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(self.camera_pitch)

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

    def _clear_old_hud_labels(self) -> None:
        """Remove labels left behind if this controller is restarted."""

        for label_id in range(12):
            self.setLabel(label_id, "", 0.0, 0.0, 0.01, HUD_TEXT)

    @staticmethod
    def print_controls() -> None:
        """Show the keyboard controls in the Webots console."""

        print("\nWildlife survey drone ready. Click the 3D view, then use:")
        print("  Up / Down                 fly forward / backward")
        print("  Left / Right              turn left / right")
        print("  Shift + Left / Right      strafe left / right")
        print("  Shift + Up / Down         raise / lower the drone")
        print("  I / K                     tilt the camera")
        print("  V                         switch camera view\n")

    def read_controls(self, now: float) -> FlightControls:
        """Read every queued key and return the requested movement."""

        # collect movement from every key in this step
        controls = FlightControls()
        camera_change = 0.0

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
            elif key in (ord("I"), ord("i")):
                camera_change -= 0.04
            elif key in (ord("K"), ord("k")):
                camera_change += 0.04
            elif key in (ord("V"), ord("v")):
                self._toggle_view(now)

            key = self.keyboard.getKey()

        # adjust the camera once after reading the keys
        if camera_change:
            self.camera_pitch = clamp(
                self.camera_pitch + camera_change,
                self.min_camera_pitch,
                self.max_camera_pitch,
            )
            self.camera_pitch_motor.setPosition(self.camera_pitch)

        return controls

    def _toggle_view(self, now: float) -> None:
        # stop a held key from switching too quickly
        if now < self.next_action_time:
            return
        self.next_action_time = now + self.action_cooldown
        self.view.toggle()
        self.next_hud_update = 0.0
        print(f"[view] {self.view.label.lower()}")

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
        """Refresh the visible-deer list when a new camera frame is ready."""

        # only scan after a new camera frame
        if now < self.next_detection_time:
            return
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

        # refresh early when the deer count changes
        if not count_changed and now < self.next_hud_update:
            return

        drone_view = self.view.mode == DRONE_VIEW
        view_text = "Drone camera - LIVE" if drone_view else "Chase view"

        # keep related labels spaced with the text scale
        title_y = 0.025
        count_y = title_y + (0.045 * HUD_TEXT_SCALE) + 0.025
        controls_second_y = 0.945
        controls_first_y = controls_second_y - (0.026 * HUD_TEXT_SCALE) - 0.012

        # draw view status and deer count
        self.set_hud_text(0, view_text, 0.025, title_y, 0.045, HUD_ACCENT)
        self.set_hud_text(
            1,
            f"Deer visible: {deer_count}",
            0.025,
            count_y,
            0.078,
            HUD_HIGHLIGHT,
        )
        # draw controls at the bottom of the screen
        self.set_hud_text(
            3,
            "Move: arrows    Strafe / altitude: Shift + arrows    Camera: I / K",
            0.025,
            controls_first_y,
            0.026,
            HUD_TEXT,
        )
        self.set_hud_text(
            4,
            "Switch view: V",
            0.025,
            controls_second_y,
            0.026,
            HUD_TEXT,
        )
        # show the reticle only in drone view
        self.set_hud_text(
            5,
            "+" if drone_view else "",
            0.487,
            0.455,
            0.072,
            HUD_HIGHLIGHT,
        )

        print(
            f"[vision] deer visible: {deer_count} | "
            f"altitude: {self.position[2]:.1f} m"
        )
        self.last_deer_count = deer_count
        self.next_hud_update = now + self.hud_update_interval

    def run(self) -> None:
        """Run one controller update for every simulation step."""

        # keep flight responsive while slower updates run
        while self.step(self.time_step) != -1:
            now = self.getTime()
            controls = self.read_controls(now)
            self.move_drone(controls)
            self.update_leds(now)
            self.view.update()
            self.update_detection(now)
            self.update_hud(now)


if __name__ == "__main__":
    # start the controller when webots runs this file
    WildlifeSurveyDrone().run()
