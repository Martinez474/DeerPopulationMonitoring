# Copyright 1996-2024 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Climb above launch, then visit absolute world XYZ waypoints in order.

Select controller "maveric_patrol" on the Mavic2Pro in Webots.
The supplied flight tuning expects WorldInfo.basicTimeStep = 8 and
WorldInfo.defaultDamping linear = 0.5, angular = 0.5.
Takeoff height is relative to launch; route altitudes are absolute world Z.
No obstacle avoidance or terrain following.
"""

import math

from controller import Robot


def clamp(value, low, high):
    return min(max(value, low), high)


class Mavic(Robot):
    K_VERTICAL_THRUST = 68.5
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0
    K_ROLL_P = 50.0
    K_PITCH_P = 30.0
    MAX_YAW_DISTURBANCE = 1.3
    K_YAW_P = 1.0
    K_YAW_D = 0.8
    MAX_PITCH_DISTURBANCE = -1.0
    TARGET_PRECISION = 2.0

    ALTITUDE_PRECISION = 2.0
    # Absolute world X, Y, Z coordinates in meters, visited once in this order.
    WAYPOINTS = (
        (34460.2, 15170.1, 382.85),
        (34142.3, 14365.7, 352.0),
    )
    ALTITUDE_ABOVE_START = 15.0

    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())
        self.camera = self.getDevice("camera")
        self.imu = self.getDevice("inertial unit")
        self.gps = self.getDevice("gps")
        self.gyro = self.getDevice("gyro")
        for sensor in (self.camera, self.imu, self.gps, self.gyro):
            sensor.enable(self.time_step)

        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)
        self.motors = [self.getDevice(name) for name in (
            "front left propeller", "front right propeller",
            "rear left propeller", "rear right propeller",
        )]
        for motor in self.motors:
            motor.setPosition(float("inf"))
            motor.setVelocity(1.0)
        self.target_index = 0

    def move_to_target(self, x, y, altitude, yaw, yaw_rate):
        target_x, target_y, target_z = self.waypoints[self.target_index]
        horizontal_reached = math.hypot(target_x - x, target_y - y) < self.TARGET_PRECISION
        altitude_reached = abs(target_z - altitude) < self.ALTITUDE_PRECISION
        if altitude_reached and (self.target_index == 0 or horizontal_reached):
            if self.target_index < len(self.waypoints) - 1:
                self.target_index += 1
                target_x, target_y, target_z = self.waypoints[self.target_index]
                print(f"Next target: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
            elif not self.route_complete:
                self.route_complete = True
                print("Final waypoint reached; maintaining the final target.")
        self.target_altitude = target_z

        # Complete the initial vertical climb before traveling toward the route.
        # At subsequent targets, stop forward input while altitude settles.
        distance = math.hypot(target_x - x, target_y - y)
        pitch_disturbance = 0.0
        if self.target_index != 0 and distance >= self.TARGET_PRECISION:
            self.target_heading = math.atan2(target_y - y, target_x - x)

        angle_left = (self.target_heading - yaw + math.pi) % (2 * math.pi) - math.pi
        # Positive mixer yaw input turns left. Oppose measured yaw rotation.
        yaw_disturbance = clamp(
            self.K_YAW_P * angle_left - self.K_YAW_D * yaw_rate,
            -self.MAX_YAW_DISTURBANCE, self.MAX_YAW_DISTURBANCE,
        )
        # Turn first; taper forward input near the target and while turning.
        if (self.target_index != 0 and distance >= self.TARGET_PRECISION
                and abs(angle_left) < 0.3):
            pitch_disturbance = (self.MAX_PITCH_DISTURBANCE
                                 * min(distance / 10.0, 1.0)
                                 * max(0.0, 1.0 - abs(angle_left) / 0.3))
        return yaw_disturbance, pitch_disturbance

    def run(self):
        # Step once so GPS reports the launch position before we create targets.
        if self.step(self.time_step) == -1:
            return
        start_x, start_y, start_z = self.gps.getValues()
        self.target_altitude = start_z + self.ALTITUDE_ABOVE_START
        self.waypoints = [(start_x, start_y, self.target_altitude), *self.WAYPOINTS]
        self.route_complete = False
        self.target_heading = self.imu.getRollPitchYaw()[2]
        print(f"Target altitude: {self.target_altitude:.2f} m "
              f"({self.ALTITUDE_ABOVE_START:.1f} m above launch)")
        print(f"Patrol waypoints: {self.waypoints}")
        print("No obstacle avoidance or terrain following.")

        last_report = self.getTime()
        yaw_disturbance = pitch_disturbance = 0.0
        while self.step(self.time_step) != -1:
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x, y, altitude = self.gps.getValues()
            roll_velocity, pitch_velocity, yaw_rate = self.gyro.getValues()

            # Run heading feedback on every physics step, including takeoff.
            yaw_disturbance, pitch_disturbance = self.move_to_target(
                x, y, altitude, yaw, yaw_rate,
            )
            if self.getTime() - last_report >= 5.0:
                tx, ty, tz = self.waypoints[self.target_index]
                print(f"Target {self.target_index}: distance={math.hypot(tx-x, ty-y):.1f} m, "
                      f"altitude={altitude:.1f}/{tz:.1f} m, "
                      f"yaw rate={math.degrees(yaw_rate):.1f} deg/s")
                last_report = self.getTime()

            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_velocity
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_velocity + pitch_disturbance
            altitude_error = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            thrust = self.K_VERTICAL_THRUST + self.K_VERTICAL_P * altitude_error ** 3
            motor_inputs = (
                thrust - yaw_disturbance + pitch_input - roll_input,
                -(thrust + yaw_disturbance + pitch_input + roll_input),
                -(thrust + yaw_disturbance - pitch_input - roll_input),
                thrust - yaw_disturbance - pitch_input + roll_input,
            )
            for motor, velocity in zip(self.motors, motor_inputs):
                motor.setVelocity(velocity)


if __name__ == "__main__":
    Mavic().run()
