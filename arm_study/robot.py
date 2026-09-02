"""
A 4-DOF arm built on the Puma 560's link geometry: kinematics, closed-form
inverse kinematics, the task Jacobian and the singularity measure.

Why 4 DOF, and how that squares with the Puma 560
-------------------------------------------------
The Puma 560 is a 6-DOF arm: three joints place the wrist centre and three
orient the tool.  This study is a top-down pick-and-place cell -- the robot
descends vertically onto a loading port on each machine's top face -- so the
tool's roll about its own approach axis is irrelevant to the task.  The two
roll joints (the manufacturer's joints 4 and 6) are therefore **locked at
zero**, leaving four driven joints:

===========  ====================================  ==========================
this model   what it does                          Puma 560 joint
===========  ====================================  ==========================
``q1``       waist: rotates the arm about the       1
             vertical
``q2``       shoulder pitch                         2
``q3``       elbow pitch                            3
``q4``       wrist pitch                            5
===========  ====================================  ==========================

The link dimensions are the published Puma 560 values (metres)::

    a2 = 0.4318   upper arm
    a3 = 0.0203   forearm offset
    d3 = 0.15005  shoulder offset, perpendicular to the arm plane
    d4 = 0.4318   forearm
    d1 = 0.6718   shoulder height above the floor

The forearm is modelled as one rigid link of length ``L3 = hypot(a3, d4) =
0.4323`` m, which is exactly the elbow-to-wrist distance of the real arm.

Exact agreement with the manufacturer's model
---------------------------------------------
The reference is the standard Denavit-Hartenberg table published for the Puma
560 (Unimation; the same table Corke's Robotics Toolbox ships as
``mdl_puma560``)::

    j   theta      d         a        alpha      travel
    1   q1         0         0        +pi/2      -160 .. +160 deg
    2   q2         0         0.4318    0         - 45 .. +225 deg
    3   q3         0.15005   0.0203   -pi/2      -225 .. + 45 deg
    4   q4         0.4318    0        +pi/2      -110 .. +170 deg
    5   q5         0         0        -pi/2      -100 .. +100 deg
    6   q6         0         0         0         -266 .. +266 deg

This model's angles differ from that table by a fixed offset per joint and by
nothing else.  :meth:`Puma560Arm.to_puma_angles` and
:meth:`Puma560Arm.from_puma_angles` are exact inverses, and the self-test
checks this model's tool pose against :func:`puma_dh_transform` -- an
independent implementation of the table above -- to machine precision::

    theta1 = q1
    theta2 = q2
    theta3 = q3 - pi/2 + psi3        psi3 = atan2(a3, d4) = 2.6926 deg
    theta4 = 0                        (locked)
    theta5 = q4 - psi3
    theta6 = 0                        (locked)

Two consequences are worth spelling out, because both were wrong in an
earlier version of this file:

* The ``q3`` offset is ``pi/2 - psi3``, not ``psi3``.  The manufacturer
  measures joint 3 to the ``d4`` axis, and ``d4`` runs *across* the elbow, so
  their ``q3 = 0`` puts the forearm nearly perpendicular to the upper arm.
  The straight-elbow pose is ``theta3 = -pi/2 + psi3``.
* The shoulder offset ``d3`` is on the **negative** side of the arm plane:
  with ``q1 = 0`` the arm reaches along ``+x`` and the shoulder sits at
  ``y = -d3``.  Getting that sign wrong mirrors the robot, which does not
  matter while the joint travel is symmetric and matters a great deal now
  that it is the manufacturer's -- asymmetric -- travel.

:data:`PUMA_TRAVEL` holds the table's travel verbatim and :class:`ArmLimits`
derives this model's ranges from it.  Nothing is rounded "inwards" any more:
a configuration this model calls legal is one the real arm can hold, and one
it calls illegal the real arm cannot.

Joint 2 travels past ``+180`` degrees, so angles can no longer simply be
folded into ``(-pi, pi]`` -- a genuine ``+200`` degree shoulder solution would
come back as ``-160`` and be rejected.  :meth:`Puma560Arm._wrap_into_travel`
folds each joint into the ``2*pi`` window centred on its own travel instead.

Task and why the arm is exactly determined
------------------------------------------
The task is the tool tip position ``(x, y, z)`` **plus** the constraint that
the tool points straight down.  That is four constraints on four joints, so a
target admits a finite set of exact solutions rather than a null space -- there
is nothing to optimise inside the IK and no iteration, which matters when the
GA calls this 60 000 times per run.  Four branches exist (shoulder left/right
x elbow up/down); :mod:`arm_study.evaluator` picks one branch and holds it for
the whole cycle, as a real cell would rather than reconfiguring mid-cycle.

Singularity
-----------
Because the vertical approach fixes the tool pitch, ``q4`` is not free: it is
whatever makes ``q2 + q3 + q4 = -pi/2``.  The motion actually available to the
tool is therefore three-dimensional, and the honest Jacobian is the 3x3
:meth:`position_jacobian` -- the derivative of tool position with respect to
``(q1, q2, q3)`` along the constraint.  Its smallest singular value
``sigma_min`` has units of metres per radian and answers the question the
threshold is really asking: how much tool motion do I get for a unit of joint
motion, in the worst direction?

Its determinant works out to ``a2 * L3 * sin(q3) * sqrt(rho^2 + d3^2)``, so the
arm is singular exactly at elbow lock-out (``q3 = 0``) and full fold
(``q3 = +-pi``).  Note the shoulder offset ``d3`` keeps the last factor away
from zero, which is precisely why the real Puma has that offset -- the
shoulder singularity over the base axis is designed out.  Requiring
``sigma_min >= threshold`` therefore carves out a well-conditioned annulus:
machines may sit neither at arm's length nor folded up against the column.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

#: The tool points straight down: pitch measured from the horizontal.
VERTICAL_PITCH = -math.pi / 2.0

#: Published Puma 560 standard-DH link geometry, in metres.
A2, A3, D3, D4 = 0.4318, 0.0203, 0.15005, 0.4318
#: Floor to shoulder axis.  Not part of the DH table -- that table starts at
#: the shoulder -- so it is carried here as the pedestal height.
D1 = 0.6718

#: Elbow-to-wrist distance, and this model's q3 zero relative to the d4 axis.
L3 = math.hypot(A3, D4)
PSI3 = math.atan2(A3, D4)

#: The manufacturer's joint travel, degrees, joints 1..6, verbatim.
PUMA_TRAVEL: Tuple[Tuple[float, float], ...] = (
    (-160.0, 160.0),
    (-45.0, 225.0),
    (-225.0, 45.0),
    (-110.0, 170.0),
    (-100.0, 100.0),
    (-266.0, 266.0),
)

#: The manufacturer's maximum joint speeds, degrees per second, joints 1..6.
PUMA_MAX_SPEED_DEG: Tuple[float, ...] = (100.0, 95.0, 100.0, 150.0, 130.0, 190.0)

#: Which manufacturer joint each of this model's four joints is, 0-based.
PUMA_JOINT_INDEX: Tuple[int, ...] = (0, 1, 2, 4)

#: Fixed offset from this model's angle to the manufacturer's, per driven
#: joint: ``theta_puma = q_model - PUMA_OFFSET``.
PUMA_OFFSET: Tuple[float, ...] = (0.0, 0.0, math.pi / 2.0 - PSI3, PSI3)

#: The two roll joints this study locks, and the value they are locked at.
LOCKED_PUMA_JOINTS: Dict[int, float] = {3: 0.0, 5: 0.0}

#: (shoulder, elbow) sign pairs.  Shoulder +1 puts the wrist on the far side
#: of the waist axis, elbow +1 bends the elbow one way; the four combinations
#: are the arm's four exact postures for a reachable target.
BRANCHES: Tuple[Tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))

BRANCH_NAMES = {
    (1, 1): "right/up",
    (1, -1): "right/down",
    (-1, 1): "left/up",
    (-1, -1): "left/down",
}


def _model_travel() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """The manufacturer's travel for the four driven joints, in model angles."""
    lower, upper = [], []
    for k, joint in enumerate(PUMA_JOINT_INDEX):
        low, high = PUMA_TRAVEL[joint]
        lower.append(math.radians(low) + PUMA_OFFSET[k])
        upper.append(math.radians(high) + PUMA_OFFSET[k])
    return tuple(lower), tuple(upper)


_Q_MIN, _Q_MAX = _model_travel()
_V_MAX = tuple(math.radians(PUMA_MAX_SPEED_DEG[j]) for j in PUMA_JOINT_INDEX)


@dataclass(frozen=True)
class ArmLimits:
    """Joint ranges, rates and accelerations, in this model's convention.

    The ranges and the rates are the Puma 560's published values mapped
    through :data:`PUMA_OFFSET`, exactly -- not rounded, not symmetrised.  In
    degrees the ranges come out as::

        q1 waist     -160.00 .. +160.00   (joint 1, no offset)
        q2 shoulder   -45.00 .. +225.00   (joint 2, no offset)
        q3 elbow     -137.69 .. +132.31   (joint 3, shifted by 90 - psi3)
        q4 wrist      -97.31 .. +102.69   (joint 5, shifted by psi3)

    Unimation never published acceleration limits, so those alone are
    representative values, and they stay parameters rather than constants so
    that the one soft number in the model is visible.
    """

    q_min: Tuple[float, ...] = _Q_MIN
    q_max: Tuple[float, ...] = _Q_MAX
    #: rad/s -- joints 1, 2, 3, 5 at their published 100, 95, 100, 130 deg/s.
    velocity: Tuple[float, ...] = _V_MAX
    #: rad/s^2 -- not published; representative.
    acceleration: Tuple[float, ...] = (6.0, 6.0, 6.0, 10.0)

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(self.q_min, dtype=float),
            np.asarray(self.q_max, dtype=float),
            np.asarray(self.velocity, dtype=float),
            np.asarray(self.acceleration, dtype=float),
        )


@dataclass
class Puma560Arm:
    """The 4-DOF arm: forward kinematics, inverse kinematics, Jacobian.

    Every method is vectorised over a leading batch axis, because the
    evaluator solves one whole cycle's worth of waypoints at a time.
    """

    base: Tuple[float, float] = (0.0, 0.0)
    d1: float = D1   # shoulder height
    a2: float = A2   # upper arm
    a3: float = A3   # forearm offset
    d3: float = D3   # shoulder offset
    d4: float = D4   # forearm
    tool: float = 0.10   # gripper: wrist centre to tool tip
    limits: ArmLimits = field(default_factory=ArmLimits)

    def __post_init__(self) -> None:
        #: Elbow-to-wrist distance of the real arm, as one rigid link.
        self.L3 = math.hypot(self.a3, self.d4)
        #: This model's q3 zero relative to the manufacturer's d4 axis.
        self.psi3 = math.atan2(self.a3, self.d4)

    # -- reach envelope ------------------------------------------------

    @property
    def max_reach(self) -> float:
        """Shoulder-to-wrist distance with the elbow straight."""
        return self.a2 + self.L3

    @property
    def min_reach(self) -> float:
        """Shoulder-to-wrist distance at the tightest permitted elbow angle."""
        q3 = min(abs(self.limits.q_min[2]), abs(self.limits.q_max[2]))
        return math.sqrt(
            self.a2**2 + self.L3**2 + 2.0 * self.a2 * self.L3 * math.cos(q3)
        )

    # -- the manufacturer's convention ---------------------------------

    def to_puma_angles(self, q: np.ndarray) -> np.ndarray:
        """This model's ``(q1, q2, q3, q4)`` as the manufacturer's six angles.

        The two locked roll joints come back as the zeros they are held at, so
        the result can be handed straight to any standard Puma 560 DH model.
        Result shape is ``q.shape[:-1] + (6,)``.
        """
        q = np.asarray(q, dtype=float)
        theta = np.zeros(q.shape[:-1] + (6,), dtype=float)
        for k, joint in enumerate(PUMA_JOINT_INDEX):
            theta[..., joint] = q[..., k] - PUMA_OFFSET[k]
        for joint, value in LOCKED_PUMA_JOINTS.items():
            theta[..., joint] = value
        return theta

    def from_puma_angles(self, theta: np.ndarray) -> np.ndarray:
        """The manufacturer's six angles as this model's four.

        The exact inverse of :meth:`to_puma_angles`; joints 4 and 6 are
        dropped, since this model cannot represent a non-zero tool roll.
        """
        theta = np.asarray(theta, dtype=float)
        return np.stack(
            [theta[..., joint] + PUMA_OFFSET[k]
             for k, joint in enumerate(PUMA_JOINT_INDEX)],
            axis=-1,
        )

    def puma_within_limits(self, q: np.ndarray) -> np.ndarray:
        """Limit test done in the manufacturer's angles, as a cross-check.

        Equivalent to :meth:`within_limits` by construction; the self-test
        runs both so that a future edit to one convention cannot silently
        drift away from the other.
        """
        theta = np.degrees(self.to_puma_angles(np.atleast_2d(q)))
        low = np.array([t[0] for t in PUMA_TRAVEL])
        high = np.array([t[1] for t in PUMA_TRAVEL])
        return np.all((theta >= low - 1e-9) & (theta <= high + 1e-9), axis=-1)

    # -- forward kinematics --------------------------------------------

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Joint angles -> task vector ``(x, y, z, pitch)``.

        ``q`` has shape ``(..., 4)``; the result has shape ``(..., 4)``.
        """
        q = np.asarray(q, dtype=float)
        q1, q2, q3, q4 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        q23 = q2 + q3
        q234 = q23 + q4
        rho = self.a2 * np.cos(q2) + self.L3 * np.cos(q23) + self.tool * np.cos(q234)
        z = (
            self.d1
            + self.a2 * np.sin(q2)
            + self.L3 * np.sin(q23)
            + self.tool * np.sin(q234)
        )
        c1, s1 = np.cos(q1), np.sin(q1)
        # The shoulder offset is on the negative side of the arm plane: at
        # q1 = 0 the arm reaches along +x and the shoulder sits at y = -d3,
        # which is what the manufacturer's DH table gives.
        x = self.base[0] + rho * c1 + self.d3 * s1
        y = self.base[1] + rho * s1 - self.d3 * c1
        return np.stack([x, y, z, q234], axis=-1)

    # -- link frames, and the polyline that hangs off them --------------

    def link_frames(self, q: np.ndarray) -> np.ndarray:
        """The five 4x4 frames the solid model is built on, for one pose.

        ===  ==================================================================
        0    pedestal: the world, translated to the base.  Fixed.
        1    waist: turned by ``q1``; its ``+z`` is the waist axis.
        2    shoulder: at ``(0, -d3, d1)`` of frame 1, pitched by ``q2``.  Its
             ``+x`` runs down the upper arm, its ``+y`` is the pitch axis.
        3    elbow: ``a2`` along frame 2's ``+x``, pitched by ``q3``.
        4    wrist: ``L3`` along frame 3's ``+x``, pitched by ``q4``.  The tool
             tip is at ``(tool, 0, 0)`` in this frame.
        ===  ==================================================================

        Every joint but the waist rotates about its frame's ``y`` axis, and a
        pitch of ``+q`` lifts the link, so the rotation is ``Ry(-q)``.
        """
        q = np.asarray(q, dtype=float).reshape(-1)[:4]

        def rot_y(angle: float) -> np.ndarray:
            c, s = math.cos(angle), math.sin(angle)
            frame = np.eye(4)
            frame[0, 0], frame[0, 2] = c, s
            frame[2, 0], frame[2, 2] = -s, c
            return frame

        def rot_z(angle: float) -> np.ndarray:
            c, s = math.cos(angle), math.sin(angle)
            frame = np.eye(4)
            frame[0, 0], frame[0, 1] = c, -s
            frame[1, 0], frame[1, 1] = s, c
            return frame

        def translate(x: float, y: float, z: float) -> np.ndarray:
            frame = np.eye(4)
            frame[:3, 3] = (x, y, z)
            return frame

        pedestal = translate(self.base[0], self.base[1], 0.0)
        waist = pedestal @ rot_z(float(q[0]))
        shoulder = waist @ translate(0.0, -self.d3, self.d1) @ rot_y(-float(q[1]))
        elbow = shoulder @ translate(self.a2, 0.0, 0.0) @ rot_y(-float(q[2]))
        wrist = elbow @ translate(self.L3, 0.0, 0.0) @ rot_y(-float(q[3]))
        return np.stack([pedestal, waist, shoulder, elbow, wrist])

    def points_along_arm(
        self, q: np.ndarray, coefficients: np.ndarray
    ) -> np.ndarray:
        """World positions of points fixed on the arm's three link axes.

        ``coefficients`` is ``(m, 3)``: how far each point sits along the upper
        arm, the forearm and the tool respectively.  A point ``s`` metres down
        the upper arm is ``(s, 0, 0)``; one ``s`` down the forearm is
        ``(a2, s, 0)``; one ``s`` down the tool is ``(a2, L3, s)``.  ``q`` is
        ``(..., 4)`` and the result is ``(..., m, 3)``.

        This is :meth:`fk` with the three link lengths made free: the same
        algebra, so a point on the axis and the tool tip can never disagree.
        It exists because the clearance test needs a few dozen points on the
        arm for every one of a few dozen configurations, and walking
        :meth:`link_frames` for each of them in Python is the one place in
        this study where that cost would show.
        """
        q = np.asarray(q, dtype=float)
        coefficients = np.asarray(coefficients, dtype=float)
        a, b, c = coefficients[:, 0], coefficients[:, 1], coefficients[:, 2]

        q1 = q[..., 0, None]
        q2 = q[..., 1, None]
        q23 = q2 + q[..., 2, None]
        q234 = q23 + q[..., 3, None]

        rho = a * np.cos(q2) + b * np.cos(q23) + c * np.cos(q234)
        z = self.d1 + a * np.sin(q2) + b * np.sin(q23) + c * np.sin(q234)
        c1, s1 = np.cos(q1), np.sin(q1)
        x = self.base[0] + rho * c1 + self.d3 * s1
        y = self.base[1] + rho * s1 - self.d3 * c1
        return np.stack([x, y, z], axis=-1)

    def joint_positions(self, q: np.ndarray) -> np.ndarray:
        """The arm's polyline for drawing: base, shoulder, elbow, wrist, tip.

        ``q`` is a single ``(4,)`` configuration; the result is ``(5, 3)``.
        Read straight off :meth:`link_frames`, so the polyline and the solid
        model in :mod:`arm_study.armmesh` can never drift apart.
        """
        frames = self.link_frames(q)
        tip = frames[4] @ np.array([self.tool, 0.0, 0.0, 1.0])
        return np.stack(
            [
                frames[0][:3, 3],   # foot of the column
                frames[2][:3, 3],   # shoulder
                frames[3][:3, 3],   # elbow
                frames[4][:3, 3],   # wrist centre
                tip[:3],
            ]
        )

    # -- inverse kinematics --------------------------------------------

    def ik_vertical(
        self, targets: np.ndarray, shoulder: int = 1, elbow: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Closed-form IK for a top-down pick at each target point.

        ``targets`` is ``(n, 3)`` of tool-tip positions.  Returns ``(q, ok)``
        where ``q`` is ``(n, 4)`` and ``ok`` is ``(n,)`` marking the targets
        this branch can actually reach.  Angles for unreachable targets are
        finite but meaningless -- always mask with ``ok``.

        Unreachable means one of two things: the target is closer to the waist
        axis than the shoulder offset ``d3`` (no ``q1`` exists), or the wrist
        centre is outside the shoulder-to-wrist annulus.
        """
        targets = np.atleast_2d(np.asarray(targets, dtype=float))
        dx = targets[:, 0] - self.base[0]
        dy = targets[:, 1] - self.base[1]
        # The tool points down, so the wrist centre sits one tool length above
        # the tip -- this is the whole reason the vertical approach makes the
        # IK closed-form.
        z_wrist = targets[:, 2] + self.tool

        radial_sq = dx * dx + dy * dy - self.d3**2
        reachable = radial_sq >= 0.0
        rho = shoulder * np.sqrt(np.maximum(radial_sq, 0.0))
        # (dx, dy) is (rho, -d3) turned by q1, so q1 = atan2(dy, dx) + atan2(d3, rho).
        q1 = np.arctan2(dy, dx) + np.arctan2(self.d3, rho)

        z_rel = z_wrist - self.d1
        distance_sq = rho * rho + z_rel * z_rel
        cos_q3 = (distance_sq - self.a2**2 - self.L3**2) / (2.0 * self.a2 * self.L3)
        reachable &= np.abs(cos_q3) <= 1.0
        q3 = elbow * np.arccos(np.clip(cos_q3, -1.0, 1.0))

        q2 = np.arctan2(z_rel, rho) - np.arctan2(
            self.L3 * np.sin(q3), self.a2 + self.L3 * np.cos(q3)
        )
        # The fourth joint is spent entirely on holding the tool vertical.
        q4 = VERTICAL_PITCH - q2 - q3

        q = self._wrap_into_travel(np.stack([q1, q2, q3, q4], axis=-1))
        return q, reachable

    def _wrap_into_travel(self, q: np.ndarray) -> np.ndarray:
        """Fold each joint into the ``2*pi`` window centred on its own travel.

        Folding everything into ``(-pi, pi]`` was safe while the travel was
        symmetric, but the shoulder's real travel runs to ``+225`` deg: a
        genuine ``+200`` deg solution would come back as ``-160`` deg and be
        rejected as out of range.  Centring the window on each joint's own
        travel picks, out of the infinitely many co-terminal angles, the one
        that joint can actually hold -- and leaves the answer unchanged
        wherever the old rule was already right.
        """
        q = np.asarray(q, dtype=float)
        centre = 0.5 * (
            np.asarray(self.limits.q_min, dtype=float)
            + np.asarray(self.limits.q_max, dtype=float)
        )
        return centre + (q - centre + math.pi) % (2.0 * math.pi) - math.pi

    def within_limits(self, q: np.ndarray) -> np.ndarray:
        """Which configurations lie inside every joint's travel."""
        q = np.atleast_2d(np.asarray(q, dtype=float))
        q_min, q_max, _, _ = self.limits.as_arrays()
        return np.all((q >= q_min) & (q <= q_max), axis=-1)

    # -- differential kinematics ---------------------------------------

    def position_jacobian(self, q: np.ndarray) -> np.ndarray:
        """d(tool position) / d(q1, q2, q3) along the vertical-approach constraint.

        ``q`` is ``(n, 4)``; the result is ``(n, 3, 3)``.  ``q4`` is excluded
        because the constraint pins it to ``-pi/2 - q2 - q3``: it is not a
        degree of freedom the task can spend, and including it would report a
        manipulability the arm does not actually have.
        """
        q = np.atleast_2d(np.asarray(q, dtype=float))
        q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2]
        c1, s1 = np.cos(q1), np.sin(q1)
        q23 = q2 + q3
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q23), np.sin(q23)

        # With the pitch pinned the tool tip is the wrist centre dropped by the
        # tool length, so the tool-length term cancels out of every derivative.
        rho = self.a2 * c2 + self.L3 * c23
        d_rho_dq2 = -self.a2 * s2 - self.L3 * s23
        d_rho_dq3 = -self.L3 * s23
        d_z_dq2 = rho
        d_z_dq3 = self.L3 * c23

        n = q.shape[0]
        jacobian = np.zeros((n, 3, 3), dtype=float)
        jacobian[:, 0, 0] = -rho * s1 + self.d3 * c1
        jacobian[:, 1, 0] = rho * c1 + self.d3 * s1
        jacobian[:, 2, 0] = 0.0
        jacobian[:, 0, 1] = d_rho_dq2 * c1
        jacobian[:, 1, 1] = d_rho_dq2 * s1
        jacobian[:, 2, 1] = d_z_dq2
        jacobian[:, 0, 2] = d_rho_dq3 * c1
        jacobian[:, 1, 2] = d_rho_dq3 * s1
        jacobian[:, 2, 2] = d_z_dq3
        return jacobian

    def task_jacobian(self, q: np.ndarray) -> np.ndarray:
        """The unconstrained 4x4 ``d(x, y, z, pitch) / dq``, for reference.

        Not used for the feasibility test -- its rows mix metres and radians,
        so its singular values have no single unit.  It is here because it is
        the object the phrase "the Jacobian of a 4-DOF arm" usually names, and
        the self-test checks it against finite differences.
        """
        q = np.atleast_2d(np.asarray(q, dtype=float))
        q1, q2, q3, q4 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        c1, s1 = np.cos(q1), np.sin(q1)
        q23, q234 = q2 + q3, q2 + q3 + q4
        c2, s2 = np.cos(q2), np.sin(q2)
        c23, s23 = np.cos(q23), np.sin(q23)
        c234, s234 = np.cos(q234), np.sin(q234)

        rho = self.a2 * c2 + self.L3 * c23 + self.tool * c234
        d_rho = np.stack(
            [
                -self.a2 * s2 - self.L3 * s23 - self.tool * s234,
                -self.L3 * s23 - self.tool * s234,
                -self.tool * s234,
            ],
            axis=-1,
        )
        d_z = np.stack(
            [rho, self.L3 * c23 + self.tool * c234, self.tool * c234], axis=-1
        )

        n = q.shape[0]
        jacobian = np.zeros((n, 4, 4), dtype=float)
        jacobian[:, 0, 0] = -rho * s1 + self.d3 * c1
        jacobian[:, 1, 0] = rho * c1 + self.d3 * s1
        for k in range(3):
            jacobian[:, 0, k + 1] = d_rho[:, k] * c1
            jacobian[:, 1, k + 1] = d_rho[:, k] * s1
            jacobian[:, 2, k + 1] = d_z[:, k]
            jacobian[:, 3, k + 1] = 1.0
        return jacobian

    def sigma_min(self, q: np.ndarray) -> np.ndarray:
        """Smallest singular value of :meth:`position_jacobian`, in m/rad.

        Zero means the tool cannot be moved at all in some direction without
        infinite joint rate -- the arm is on a singularity.  One SVD call
        handles the whole stack of waypoints.
        """
        jacobian = self.position_jacobian(q)
        return np.linalg.svd(jacobian, compute_uv=False)[:, -1]

    def manipulability(self, q: np.ndarray) -> np.ndarray:
        """Yoshikawa's measure ``sqrt(det(J J^T))``, reported alongside sigma_min."""
        jacobian = self.position_jacobian(q)
        return np.abs(np.linalg.det(jacobian))


#: The standard-DH table of the Puma 560: ``(d, a, alpha)`` for joints 1..6.
PUMA_DH: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.0, math.pi / 2.0),
    (0.0, A2, 0.0),
    (D3, A3, -math.pi / 2.0),
    (D4, 0.0, math.pi / 2.0),
    (0.0, 0.0, -math.pi / 2.0),
    (0.0, 0.0, 0.0),
)


def puma_dh_transform(theta: np.ndarray, d1: float = D1) -> np.ndarray:
    """Tool-flange pose from the manufacturer's six angles, by the DH table.

    An independent implementation of the published chain, kept here so the
    self-test can check :class:`Puma560Arm` against the manufacturer's model
    rather than against itself.  ``d1`` lifts the DH origin, which sits at the
    shoulder, onto the pedestal.
    """
    pose = np.eye(4)
    pose[2, 3] = d1
    for angle, (d, a, alpha) in zip(np.asarray(theta, dtype=float).reshape(-1), PUMA_DH):
        ct, st = math.cos(float(angle)), math.sin(float(angle))
        ca, sa = math.cos(alpha), math.sin(alpha)
        pose = pose @ np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    return pose
