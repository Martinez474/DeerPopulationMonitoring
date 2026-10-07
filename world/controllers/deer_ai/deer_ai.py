"""
Simple Deer AI
"""

import math
import random
from pathlib import Path

from controller import Supervisor


# these values describe the elevation grid used to build the island
GRID_WIDTH = 512
GRID_HEIGHT = 512
X_SPACING = 90.783197
Y_SPACING = 75.587695

# heights at or below this level are treated as water instead of walkable land
LAND_HEIGHT = 1.0

# this gap keeps the lowest part of the deer just above the terrain surface
GROUND_CLEARANCE = 0.065

# deer reject terrain that is steeper than this angle
MAX_TERRAIN_SLOPE = math.radians(32.0)


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_angle(angle):
    # convert any angle to the range from negative pi to positive pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def smoothstep(value):
    # create a smooth zero-to-one blend with gentle starts and stops
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def multiply_quaternions(left, right):
    # combine two rotations without depending on euler angle order
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def quaternion_for_axis_angle(axis, angle):
    # turn a webots-style axis and angle into a quaternion
    half_angle = 0.5 * angle
    sine = math.sin(half_angle)
    return (math.cos(half_angle), axis[0] * sine, axis[1] * sine, axis[2] * sine)


def quaternion_to_axis_angle(quaternion):
    # turn the combined quaternion back into the format webots expects
    w, x, y, z = quaternion

    # normalize the quaternion
    length = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / length, x / length, y / length, z / length
    w = clamp(w, -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    sine = math.sqrt(max(0.0, 1.0 - w * w))

    # use a harmless default axis when the rotation is almost zero
    if sine < 1e-7:
        return [0.0, 0.0, 1.0, 0.0]
    return [x / sine, y / sine, z / sine, angle]


def body_rotation(yaw, pitch, roll):
    # yaw turns left and right, pitch leans uphill, and roll follows side slopes
    yaw_q = quaternion_for_axis_angle((0.0, 0.0, 1.0), yaw)
    pitch_q = quaternion_for_axis_angle((0.0, 1.0, 0.0), pitch)
    roll_q = quaternion_for_axis_angle((1.0, 0.0, 0.0), roll)

    # apply yaw first, then pitch, then roll, and return one final rotation
    return quaternion_to_axis_angle(
        multiply_quaternions(multiply_quaternions(yaw_q, pitch_q), roll_q)
    )


def load_terrain_heights():
    # find the terrain proto relative to this controller file
    project_root = Path(__file__).resolve().parents[2]
    terrain_file = project_root / "protos" / "CatalinaTerrain.proto"
    text = terrain_file.read_text(encoding="utf-8")

    # read the numbers inside the first height field in the proto
    marker = "height ["
    start = text.index(marker) + len(marker)
    end = text.index("]", start)
    heights = [float(value) for value in text[start:end].split()]

    # stop with a useful error if the terrain file has an unexpected size
    expected = GRID_WIDTH * GRID_HEIGHT
    if len(heights) != expected:
        raise RuntimeError(
            f"Expected {expected} Catalina terrain heights, found {len(heights)}"
        )
    return heights


class DeerController:
    # each state lasts for a random number of seconds inside these ranges
    DURATIONS = {
        "idle": (4.0, 9.0),
        "graze": (6.0, 14.0),
        "alert": (2.5, 5.0),
        "walk": (8.0, 18.0),
    }

    def __init__(self):
        # connect this python controller to its deer robot in webots
        self.robot = Supervisor()
        self.time_step = int(self.robot.getBasicTimeStep())

        # delta time
        self.dt = self.time_step / 1000.0

        # keep direct access to the fields that move and identify the deer
        self.node = self.robot.getSelf()
        self.translation = self.node.getField("translation")
        self.rotation = self.node.getField("rotation")
        self.name = self.node.getField("name").getSFString()
        self.custom_data = self.node.getField("customData").getSFString()

        # each deer has a different seed, so the herd does not move in sync
        try:
            seed = int(self.custom_data)
        except ValueError:
            seed = sum(ord(character) for character in self.custom_data)
        self.random = random.Random(seed)

        # load the island surface and remember this deer's starting location
        self.heights = load_terrain_heights()
        initial_position = self.translation.getSFVec3f()
        self.x = initial_position[0]
        self.y = initial_position[1]
        self.anchor_x = self.x
        self.anchor_y = self.y

        # use the starting z-axis rotation as the initial facing direction
        initial_rotation = self.rotation.getSFRotation()
        self.heading = initial_rotation[3] if abs(initial_rotation[2]) > 0.9 else 0.0
        self.desired_heading = self.heading

        # movement begins stopped and stays near the original spawn point
        self.speed = 0.0
        self.wander_radius = 140.0
        self.next_steer_time = 0.0

        # start in the idle state and choose a random idle duration
        self.state = "idle"
        self.state_started = self.robot.getTime()
        self.state_ends = self.state_started
        self.enter_state("idle")
        print(f"{self.name}: simple deer AI active")

    def terrain_height(self, x, y):
        # convert world coordinates into decimal elevation-grid coordinates
        grid_x = clamp(x / X_SPACING, 0.0, GRID_WIDTH - 1.001)
        grid_y = clamp(y / Y_SPACING, 0.0, GRID_HEIGHT - 1.001)

        # find the four terrain vertices surrounding this world position
        x0 = int(math.floor(grid_x))
        y0 = int(math.floor(grid_y))
        x1 = min(x0 + 1, GRID_WIDTH - 1)
        y1 = min(y0 + 1, GRID_HEIGHT - 1)
        tx = grid_x - x0
        ty = grid_y - y0

        # look up the height stored at each corner of the current cell
        h00 = self.heights[y0 * GRID_WIDTH + x0]
        h10 = self.heights[y0 * GRID_WIDTH + x1]
        h01 = self.heights[y1 * GRID_WIDTH + x0]
        h11 = self.heights[y1 * GRID_WIDTH + x1]

        # webots renders each elevation grid cell as two flat triangles. the
        # visible mesh uses the diagonal between h00 and h11, so the chosen
        # plane depends on which side of that diagonal contains the deer
        if tx >= ty:
            return h00 + (h10 - h00) * tx + (h11 - h10) * ty
        return h00 + (h11 - h01) * tx + (h01 - h00) * ty

    def position_is_walkable(self, x, y, heading):
        # keep the deer away from the outer edge of the elevation grid
        if not (
            X_SPACING < x < (GRID_WIDTH - 2) * X_SPACING
            and Y_SPACING < y < (GRID_HEIGHT - 2) * Y_SPACING
        ):
            return False

        # sample the ground at the center, front, back, left, and right
        sample_distance = 1.2
        forward_x = math.cos(heading) * sample_distance
        forward_y = math.sin(heading) * sample_distance
        left_x = -math.sin(heading) * sample_distance
        left_y = math.cos(heading) * sample_distance
        samples = (
            self.terrain_height(x, y),
            self.terrain_height(x + forward_x, y + forward_y),
            self.terrain_height(x - forward_x, y - forward_y),
            self.terrain_height(x + left_x, y + left_y),
            self.terrain_height(x - left_x, y - left_y),
        )

        # stop movement if any part of the deer reaches water
        if min(samples) <= LAND_HEIGHT:
            return False

        # compare opposite samples to estimate forward and sideways steepness
        max_rise = math.tan(MAX_TERRAIN_SLOPE) * (2.0 * sample_distance)
        forward_rise = abs(samples[1] - samples[2])
        sideways_rise = abs(samples[3] - samples[4])
        return forward_rise <= max_rise and sideways_rise <= max_rise

    def enter_state(self, state):
        # record the new behavior and choose when it should finish
        now = self.robot.getTime()
        self.state = state
        self.state_started = now
        low, high = self.DURATIONS[state]
        self.state_ends = now + self.random.uniform(low, high)

        # a walking state needs a safe direction before movement begins
        if state == "walk":
            self.choose_walk_heading(now)

    def choose_next_state(self):
        choice = self.random.random()

        # walking is normally followed by resting or grazing
        if self.state == "walk":
            return "graze" if choice < 0.42 else "idle"

        # grazing usually leads back into a walk
        if self.state == "graze":
            return "walk" if choice < 0.58 else "idle"

        # an alert deer usually decides to move away afterward
        if self.state == "alert":
            return "walk" if choice < 0.7 else "idle"

        if choice < 0.43:
            return "walk"
        if choice < 0.82:
            return "graze"
        return "alert"

    def choose_walk_heading(self, now):
        # measure how far the deer has wandered from its spawn point
        distance_from_anchor = math.hypot(
            self.x - self.anchor_x, self.y - self.anchor_y
        )

        # near the wander limit, turn roughly back toward the spawn point
        if distance_from_anchor > self.wander_radius * 0.72:
            center_heading = math.atan2(
                self.anchor_y - self.y, self.anchor_x - self.x
            )
            candidate = center_heading + self.random.uniform(-0.35, 0.35)
        else:
            # otherwise, make a random turn from the current direction
            candidate = self.heading + self.random.uniform(-0.9, 0.9)

        # avoid headings that lead out of the terrain or toward the sea
        for _ in range(10):
            # test a point well ahead so the deer turns before reaching danger
            probe_x = self.x + 35.0 * math.cos(candidate)
            probe_y = self.y + 35.0 * math.sin(candidate)
            if self.position_is_walkable(probe_x, probe_y, candidate):
                break

            # if the route is unsafe, try another direction toward the anchor
            candidate = math.atan2(
                self.anchor_y - self.y, self.anchor_x - self.x
            ) + self.random.uniform(-0.35, 0.35)

        # save the direction and wait a few seconds before steering again
        self.desired_heading = wrap_angle(candidate)
        self.next_steer_time = now + self.random.uniform(4.0, 8.0)

    def update_behavior(self, now):
        # change behavior when the current state's timer expires
        if now >= self.state_ends:
            self.enter_state(self.choose_next_state())

        # walking deer occasionally adjust their direction
        if self.state == "walk" and now >= self.next_steer_time:
            self.choose_walk_heading(now)

        # smoothly accelerate while walking and smoothly stop in other states
        target_speed = 0.62 if self.state == "walk" else 0.0
        acceleration = 0.38 if target_speed > self.speed else 0.6
        speed_change = clamp(
            target_speed - self.speed,
            -acceleration * self.dt,
            acceleration * self.dt,
        )
        self.speed += speed_change

        # rotate toward the desired heading at a limited, natural turn speed
        if self.state == "walk":
            heading_error = wrap_angle(self.desired_heading - self.heading)
            self.heading = wrap_angle(
                self.heading + clamp(heading_error, -0.32 * self.dt, 0.32 * self.dt)
            )

        # predict the next horizontal position before committing to the move
        next_x = self.x + math.cos(self.heading) * self.speed * self.dt
        next_y = self.y + math.sin(self.heading) * self.speed * self.dt

        # stop and turn around when the next step is water, an edge, or too steep
        if not self.position_is_walkable(next_x, next_y, self.heading):
            self.desired_heading = wrap_angle(self.heading + math.pi)
            self.speed = 0.0
        else:
            self.x = next_x
            self.y = next_y

    def body_gesture(self, now):
        # measure progress through the current behavior state
        elapsed = now - self.state_started
        remaining = self.state_ends - now

        # every return value contains height, pitch, roll, and extra yaw
        if self.state == "walk":
            # walking adds a repeating vertical bob and side-to-side sway
            stride = 2.0 * math.pi * 1.55 * elapsed
            return (
                0.018 + 0.022 * abs(math.sin(stride)),
                0.018 * math.sin(stride),
                0.025 * math.sin(stride + math.pi / 2.0),
                0.0,
            )
        if self.state == "graze":
            # grazing slowly lowers and pitches the body, then raises it again
            blend = smoothstep(min(elapsed / 1.4, remaining / 1.4))
            return (
                -0.035 * blend + 0.006 * math.sin(now * 1.2),
                0.12 * blend,
                0.012 * math.sin(now * 0.7),
                0.0,
            )
        if self.state == "alert":
            # alert behavior raises the body and scans left and right
            blend = smoothstep(min(elapsed / 0.6, remaining / 0.8))
            return (
                0.018 * blend,
                -0.035 * blend,
                0.008 * math.sin(now * 1.8),
                0.075 * math.sin(elapsed * 1.1) * blend,
            )

        # idle behavior uses tiny breathing and balance motions
        return (
            0.006 * math.sin(now * 1.05),
            0.008 * math.sin(now * 0.55),
            0.012 * math.sin(now * 0.42),
            0.0,
        )

    def update_pose(self, now):
        # get the small animation offsets for the current behavior
        body_height, gesture_pitch, roll, yaw_offset = self.body_gesture(now)

        # build forward and left sample directions from the current heading
        sample_distance = 1.0
        forward_x = math.cos(self.heading) * sample_distance
        forward_y = math.sin(self.heading) * sample_distance
        left_x = -math.sin(self.heading) * sample_distance
        left_y = math.cos(self.heading) * sample_distance

        # read terrain heights around the deer's footprint
        height_ahead = self.terrain_height(self.x + forward_x, self.y + forward_y)
        height_behind = self.terrain_height(self.x - forward_x, self.y - forward_y)
        height_left = self.terrain_height(self.x + left_x, self.y + left_y)
        height_right = self.terrain_height(self.x - left_x, self.y - left_y)

        # pitch the body to follow uphill and downhill terrain
        terrain_pitch = -math.atan2(
            height_ahead - height_behind, 2.0 * sample_distance
        )
        terrain_pitch = clamp(terrain_pitch, -0.3, 0.3)

        # roll the body so its left and right sides match the side slope
        terrain_roll = math.atan2(
            height_left - height_right, 2.0 * sample_distance
        )
        terrain_roll = clamp(terrain_roll, -0.3, 0.3)

        # place the deer directly above the current terrain surface
        ground_height = self.terrain_height(self.x, self.y)
        self.translation.setSFVec3f(
            [self.x, self.y, ground_height + GROUND_CLEARANCE + body_height]
        )

        # combine travel direction, terrain tilt, and behavior animation
        self.rotation.setSFRotation(
            body_rotation(
                self.heading + yaw_offset,
                terrain_pitch + gesture_pitch,
                terrain_roll + roll,
            )
        )

    def run(self):
        # update movement and pose once for every webots simulation step
        while self.robot.step(self.time_step) != -1:
            now = self.robot.getTime()
            self.update_behavior(now)
            self.update_pose(now)


# create the controller and keep it running until the simulation stops
DeerController().run()
