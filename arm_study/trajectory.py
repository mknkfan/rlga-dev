"""
Point-to-point motion with a trapezoidal velocity profile, and the cycle time
that falls out of it.

The model
--------
Between two stations the arm makes a coordinated joint-space move: every joint
starts together, stops together, and follows the straight line between the two
configurations.  Writing the move as ``q(t) = q_start + s(t) * dq`` with a
single path parameter ``s`` running 0 -> 1, the whole motion is described by
one scalar profile, and ``s`` is given the classic trapezoid -- accelerate at a
constant rate, cruise at a constant rate, decelerate at the mirror of the
acceleration.  A short move never reaches cruise and the trapezoid degenerates
to a triangle; both cases are handled.

Why a shared ``s`` rather than a profile per joint: it keeps the joints
synchronised, so the tool follows a repeatable path instead of a shape that
depends on which joint happened to finish first.  This is the standard
trapezoidal time scaling (Lynch & Park, *Modern Robotics*, sec. 9.2).

Respecting the limits
---------------------
Joint ``i`` moves through ``dq_i``, so its rate is ``|dq_i| * s_dot`` and its
acceleration ``|dq_i| * s_ddot``.  Every joint therefore stays inside its own
limits as long as

    s_dot  <= min_i ( v_i / |dq_i| )        and
    s_ddot <= min_i ( a_i / |dq_i| )

and the fastest legal move takes those bounds with equality.  The joint that
attains the minimum is the one governing the move; the others coast below
their limits.  This is what makes the cycle time a real function of the layout
rather than a rescaled distance: a move that swings the waist a long way is
paced by the waist, one that mostly reaches out is paced by the shoulder or
elbow, and the two have different limits.

Cycle
-----
The robot starts at a home pose, visits the machines in the process order, and
returns home.  Cycle time is the sum of the segment times plus a fixed dwell
at each station for the actual pick or place.  Dwell is a constant, so it
shifts every layout equally and never changes the ranking -- it is included so
the reported number is a cycle time rather than a travel time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: Moves shorter than this (in radians, on every joint) are treated as no move.
_NEGLIGIBLE = 1e-12


@dataclass(frozen=True)
class SegmentProfile:
    """The trapezoid for one point-to-point move.

    ``duration`` is the move time; ``peak_rate`` and ``rate_limit`` are in
    units of the path parameter per second and per second squared.
    ``triangular`` records whether the move was too short to reach cruise.
    """

    duration: float
    peak_rate: float
    acceleration: float
    triangular: bool
    delta: np.ndarray

    def path_parameter(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(s, s_dot)`` at times ``t``, clamped to the segment."""
        t = np.clip(np.asarray(t, dtype=float), 0.0, self.duration)
        if self.duration <= 0.0:
            return np.zeros_like(t), np.zeros_like(t)

        a = self.acceleration
        s = np.empty_like(t)
        s_dot = np.empty_like(t)

        if self.triangular:
            half = self.duration / 2.0
            rising = t <= half
            s[rising] = 0.5 * a * t[rising] ** 2
            s_dot[rising] = a * t[rising]
            falling = ~rising
            remaining = self.duration - t[falling]
            s[falling] = 1.0 - 0.5 * a * remaining**2
            s_dot[falling] = a * remaining
        else:
            v = self.peak_rate
            ramp = v / a
            accelerating = t < ramp
            decelerating = t > self.duration - ramp
            cruising = ~(accelerating | decelerating)

            s[accelerating] = 0.5 * a * t[accelerating] ** 2
            s_dot[accelerating] = a * t[accelerating]

            s[cruising] = v * t[cruising] - v * v / (2.0 * a)
            s_dot[cruising] = v

            remaining = self.duration - t[decelerating]
            s[decelerating] = 1.0 - 0.5 * a * remaining**2
            s_dot[decelerating] = a * remaining

        return s, s_dot

    def sample(
        self, q_start: np.ndarray, samples: int = 60
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(t, q, q_dot)`` across the segment, for plotting and animation."""
        t = np.linspace(0.0, self.duration, max(samples, 2))
        s, s_dot = self.path_parameter(t)
        q = q_start[None, :] + s[:, None] * self.delta[None, :]
        q_dot = s_dot[:, None] * self.delta[None, :]
        return t, q, q_dot


def plan_segment(
    q_start: np.ndarray,
    q_end: np.ndarray,
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
) -> SegmentProfile:
    """The time-optimal trapezoidal move between two configurations."""
    delta = np.asarray(q_end, dtype=float) - np.asarray(q_start, dtype=float)
    travel = np.abs(delta)
    moving = travel > _NEGLIGIBLE
    if not moving.any():
        return SegmentProfile(0.0, 0.0, 0.0, True, delta)

    # The governing joint is whichever is first to hit a limit.
    rate_cap = float(np.min(velocity_limit[moving] / travel[moving]))
    accel_cap = float(np.min(acceleration_limit[moving] / travel[moving]))

    if rate_cap**2 / accel_cap >= 1.0:
        # Too short to reach cruise: accelerate to the midpoint, then brake.
        duration = 2.0 * math.sqrt(1.0 / accel_cap)
        # The trapezoid collapses; the effective rate follows from s(T/2)=1/2.
        return SegmentProfile(duration, accel_cap * duration / 2.0, 4.0 / duration**2, True, delta)

    duration = 1.0 / rate_cap + rate_cap / accel_cap
    return SegmentProfile(duration, rate_cap, accel_cap, False, delta)


def plan_cycle(
    configurations: Sequence[np.ndarray],
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
) -> List[SegmentProfile]:
    """One profile per consecutive pair in the visiting order."""
    return [
        plan_segment(configurations[k], configurations[k + 1], velocity_limit, acceleration_limit)
        for k in range(len(configurations) - 1)
    ]


def dwell_indices(
    configurations: Sequence[np.ndarray], dwell_at: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Which waypoints the arm actually holds at.

    Without a retract the tour is home -> station -> ... -> home and every
    interior waypoint is a station, which is the default.  With one, the tour
    also carries a lift pose above each port on the way in and on the way out,
    and those are passed through rather than held at -- so the caller has to
    say which is which or the cycle picks up two spurious dwells per station.
    """
    if dwell_at is not None:
        return np.asarray(dwell_at, dtype=int)
    return np.arange(1, max(len(configurations) - 1, 1))


def cycle_time(
    configurations: Sequence[np.ndarray],
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    dwell: float = 0.0,
    dwell_at: Optional[Sequence[int]] = None,
) -> float:
    """Total time for home -> every station -> home, including the dwells.

    ``configurations`` is the whole tour, home at both ends.  ``dwell_at``
    names the waypoints that are stations; leaving it out means every interior
    waypoint is one.
    """
    segments = plan_cycle(configurations, velocity_limit, acceleration_limit)
    moving = sum(segment.duration for segment in segments)
    stations = len(dwell_indices(configurations, dwell_at))
    return float(moving + dwell * stations)


def sample_cycle(
    configurations: Sequence[np.ndarray],
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    dwell: float = 0.0,
    samples_per_segment: int = 60,
    dwell_at: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The whole cycle as ``(t, q, q_dot)`` on one clock.

    Dwells appear as flat stretches, which is what makes the joint-angle plot
    read as a cycle rather than a sequence of unrelated moves.
    """
    segments = plan_cycle(configurations, velocity_limit, acceleration_limit)
    holds = set(int(k) for k in dwell_indices(configurations, dwell_at))
    times: List[np.ndarray] = []
    angles: List[np.ndarray] = []
    rates: List[np.ndarray] = []

    clock = 0.0
    for index, segment in enumerate(segments):
        t, q, q_dot = segment.sample(np.asarray(configurations[index], dtype=float),
                                     samples_per_segment)
        times.append(t + clock)
        angles.append(q)
        rates.append(q_dot)
        clock += segment.duration

        # A dwell on arriving at a station -- not at a lift pose, and not on
        # the final return home.
        if dwell > 0.0 and (index + 1) in holds:
            hold = np.asarray(configurations[index + 1], dtype=float)
            times.append(np.array([clock, clock + dwell]))
            angles.append(np.stack([hold, hold]))
            rates.append(np.zeros((2, hold.size)))
            clock += dwell

    return (
        np.concatenate(times),
        np.concatenate(angles, axis=0),
        np.concatenate(rates, axis=0),
    )
