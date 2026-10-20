"""Extensible behavior-state classes for Catalina deer."""

from abc import ABC, abstractmethod
import math
from typing import Any


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class DeerState(ABC):
    """Base class for one deer behavior.

    New states only need to define their duration, transition decision, and
    optional movement update. Flight and terrain code remain untouched.
    """

    name = "base"
    duration_range = (1.0, 1.0)
    target_speed = 0.0

    def enter(self, deer: Any, now: float) -> None:
        """Run state-specific setup when this state becomes active."""

    def update(self, deer: Any, now: float) -> None:
        """Update state-specific behavior before movement is applied."""

    @abstractmethod
    def choose_next(self, deer: Any) -> "DeerState":
        """Choose and construct the next behavior state."""


class IdleState(DeerState):
    name = "idle"
    duration_range = (4.0, 9.0)

    def choose_next(self, deer: Any) -> DeerState:
        choice = deer.random.random()
        if choice < 0.43:
            return WalkState()
        if choice < 0.82:
            return GrazeState()
        return AlertState()


class GrazeState(DeerState):
    name = "graze"
    duration_range = (6.0, 14.0)

    def choose_next(self, deer: Any) -> DeerState:
        return WalkState() if deer.random.random() < 0.58 else IdleState()


class AlertState(DeerState):
    name = "alert"
    duration_range = (2.5, 5.0)

    def choose_next(self, deer: Any) -> DeerState:
        return WalkState() if deer.random.random() < 0.7 else IdleState()


class WalkState(DeerState):
    name = "walk"
    duration_range = (8.0, 18.0)
    target_speed = 0.62

    def enter(self, deer: Any, now: float) -> None:
        deer.choose_walk_heading(now)

    def update(self, deer: Any, now: float) -> None:
        if now >= deer.next_steer_time:
            deer.choose_walk_heading(now)

        heading_error = _wrap_angle(deer.desired_heading - deer.heading)
        deer.heading = _wrap_angle(
            deer.heading
            + _clamp(heading_error, -0.32 * deer.dt, 0.32 * deer.dt)
        )

    def choose_next(self, deer: Any) -> DeerState:
        return GrazeState() if deer.random.random() < 0.42 else IdleState()
