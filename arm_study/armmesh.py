"""
A solid model of the Puma 560, built from the same link frames the kinematics
use.

Why this exists
---------------
The arm used to be drawn as a five-point polyline.  A polyline gives the eye
no occlusion cue and no sense of which way a link is facing, so a pose that is
reaching *away* from the camera looks identical to one reaching towards it,
and the arm never reads as an object standing among the machine boxes.  This
module puts a shape on each link: a pedestal, a rotating turret, the shoulder
and elbow castings, the two arm links as flattened tapered sections, the wrist
yoke and a two-finger gripper.

Nothing here invents geometry that the kinematics do not already contain.
Every part is expressed in one of the five frames returned by
:meth:`arm_study.robot.Puma560Arm.link_frames`, so the solid model and the
maths are the same object: the upper arm is drawn from the shoulder frame's
origin to ``x = a2``, the forearm from the elbow frame's origin to ``x = L3``,
and the gripper fingers end exactly at ``x = tool`` in the wrist frame, which
is the tool tip the IK solves for.  Change a link length in ``robot.py`` and
the picture follows.

How it is drawn
---------------
Each part is a list of convex polygons in its link's local frame.  Two
consumers use them:

* :mod:`arm_study.visualize` transforms them to world coordinates and hands
  the whole scene -- machines *and* arm -- to a single matplotlib
  ``Poly3DCollection``.  One collection is the point: matplotlib depth-sorts
  within a collection but not between collections, so putting everything in
  one is what makes the arm occlude and be occluded by the machines correctly.
* :mod:`arm_study.viewer` ships the local polygons to the browser **once**,
  with only the four joint angles per frame, and rebuilds the transforms in
  JavaScript.  That is both smaller than sending baked world geometry per
  frame and what lets the page re-shade the arm as the viewpoint turns.

Shading is one directional light plus an ambient term, applied per polygon
from its world normal.  It is not a light model anyone should believe, but it
is enough for the eye to read a cylinder as round and to tell the top of a
link from its side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from robot import Puma560Arm

#: Unimation orange for the links, dark castings at the joints, grey metal for
#: the pedestal and the gripper.  Chosen to stay legible on the white
#: matplotlib page and on the viewer's near-black background alike.
ORANGE = (0.918, 0.522, 0.129)
CASTING = (0.243, 0.267, 0.298)
PEDESTAL = (0.400, 0.435, 0.478)
STEEL = (0.639, 0.678, 0.718)

#: Direction the light comes *from*, in world coordinates, and the fraction of
#: full brightness a polygon keeps when it faces away from it.
LIGHT = np.array([-0.35, -0.62, 0.70])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
AMBIENT = 0.42


@dataclass(frozen=True)
class Part:
    """One rigid piece of the robot, in the local frame of ``frame``."""

    name: str
    frame: int
    colour: Tuple[float, float, float]
    #: Convex polygons, each ``(k, 3)``, wound consistently outwards.
    polys: Tuple[np.ndarray, ...]


# ----------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------


def sweep(
    start: Sequence[float],
    end: Sequence[float],
    u: Sequence[float],
    v: Sequence[float],
    section_start: Tuple[float, float],
    section_end: Tuple[float, float],
    sides: int = 14,
    caps: bool = True,
) -> List[np.ndarray]:
    """A tapered tube from ``start`` to ``end`` with an elliptical section.

    ``u`` and ``v`` are the two axes of the section, ``section_*`` its
    semi-axes along them at each end.  Equal semi-axes give a cylinder;
    unequal ones give the flattened castings the real arm's links are.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    ring0 = start + np.outer(cos * section_start[0], u) + np.outer(sin * section_start[1], v)
    ring1 = end + np.outer(cos * section_end[0], u) + np.outer(sin * section_end[1], v)

    polys = [
        np.stack([ring0[k], ring0[(k + 1) % sides], ring1[(k + 1) % sides], ring1[k]])
        for k in range(sides)
    ]
    if caps:
        polys.append(ring0[::-1].copy())
        polys.append(ring1.copy())
    return polys


def box(
    lower: Sequence[float], upper: Sequence[float]
) -> List[np.ndarray]:
    """An axis-aligned box in the local frame, as six quads."""
    x0, y0, z0 = (float(c) for c in lower)
    x1, y1, z1 = (float(c) for c in upper)
    return [
        np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]),  # +z
        np.array([[x0, y1, z0], [x1, y1, z0], [x1, y0, z0], [x0, y0, z0]]),  # -z
        np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]),  # +x
        np.array([[x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1]]),  # -x
        np.array([[x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [x1, y1, z0]]),  # +y
        np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]),  # -y
    ]


_X = (1.0, 0.0, 0.0)
_Y = (0.0, 1.0, 0.0)
_Z = (0.0, 0.0, 1.0)


# ----------------------------------------------------------------------
# the robot
# ----------------------------------------------------------------------


def build_parts(arm: Puma560Arm, sides: int = 14) -> List[Part]:
    """The whole arm as parts, each in its link frame.

    Frames are the ones :meth:`Puma560Arm.link_frames` returns: 0 pedestal,
    1 waist, 2 upper arm, 3 forearm, 4 wrist.  ``sides`` trades polygon count
    for roundness -- 14 is about where a link stops looking faceted at figure
    resolution, and the whole robot is then ~170 polygons.
    """
    d1, a2, d3, tool = arm.d1, arm.a2, arm.d3, arm.tool
    l3 = arm.L3
    shoulder_top = d1 - 0.155
    parts: List[Part] = []

    # -- frame 0: the pedestal, which does not move ---------------------
    parts.append(Part("foot", 0, PEDESTAL, tuple(
        sweep((0, 0, 0.0), (0, 0, 0.04), _X, _Y, (0.175, 0.175), (0.150, 0.150), sides + 2)
    )))
    parts.append(Part("column", 0, PEDESTAL, tuple(
        sweep((0, 0, 0.04), (0, 0, shoulder_top), _X, _Y, (0.100, 0.100), (0.092, 0.092), sides + 2)
    )))

    # -- frame 1: the turret and the shoulder casting --------------------
    parts.append(Part("turret", 1, ORANGE, tuple(
        sweep((0, 0, shoulder_top), (0, 0, d1 - 0.005), _X, _Y,
              (0.118, 0.118), (0.106, 0.106), sides + 2)
    )))
    # The shoulder yoke lies along the pitch axis and reaches across the d3
    # offset, so it visibly ties the turret to the arm plane -- which is what
    # the offset physically is on the real machine.
    parts.append(Part("shoulder", 1, CASTING, tuple(
        sweep((0, -d3 - 0.088, d1), (0, 0.048, d1), _X, _Z, (0.094, 0.094), (0.094, 0.094), sides)
    )))

    # -- frame 2: the upper arm, +x from the shoulder to the elbow -------
    parts.append(Part("upper_arm", 2, ORANGE, tuple(
        sweep((0.0, 0, 0), (a2, 0, 0), _Y, _Z, (0.074, 0.084), (0.052, 0.062), sides)
    )))

    # -- frame 3: the elbow casting and the forearm ----------------------
    parts.append(Part("elbow", 3, CASTING, tuple(
        sweep((0, -0.084, 0), (0, 0.084, 0), _X, _Z, (0.080, 0.080), (0.080, 0.080), sides)
    )))
    parts.append(Part("forearm", 3, ORANGE, tuple(
        sweep((0.02, 0, 0), (l3 - 0.03, 0, 0), _Y, _Z, (0.058, 0.066), (0.040, 0.045), sides)
    )))

    # -- frame 4: the wrist and the gripper ------------------------------
    parts.append(Part("wrist", 4, CASTING, tuple(
        sweep((0, -0.052, 0), (0, 0.052, 0), _X, _Z, (0.050, 0.050), (0.050, 0.050), sides - 2)
    )))
    parts.append(Part("flange", 4, STEEL, tuple(
        sweep((0.0, 0, 0), (0.034, 0, 0), _Y, _Z, (0.040, 0.040), (0.036, 0.036), sides - 2)
    )))
    parts.append(Part("palm", 4, CASTING, tuple(
        box((0.034, -0.030, -0.021), (0.060, 0.030, 0.021))
    )))
    # The fingers close on the tool tip: their far face is at x = tool, which
    # is the point ik_vertical() puts on the loading port.
    for sign, name in ((1.0, "finger_a"), (-1.0, "finger_b")):
        parts.append(Part(name, 4, STEEL, tuple(
            box((0.060, sign * 0.019 - 0.008, -0.016), (tool, sign * 0.019 + 0.008, 0.016))
        )))
    return parts


# ----------------------------------------------------------------------
# to world coordinates, shaded
# ----------------------------------------------------------------------


def shade(colour: Sequence[float], normal: np.ndarray) -> np.ndarray:
    """One directional light plus ambient, from a polygon's world normal."""
    norm = float(np.linalg.norm(normal))
    lambert = 1.0 if norm < 1e-12 else abs(float(np.dot(normal / norm, LIGHT)))
    return np.clip(np.asarray(colour, dtype=float) * (AMBIENT + (1.0 - AMBIENT) * lambert), 0.0, 1.0)


def _normal(poly: np.ndarray) -> np.ndarray:
    return np.cross(poly[1] - poly[0], poly[2] - poly[0])


def posed_polygons(
    arm: Puma560Arm,
    q: np.ndarray,
    parts: Sequence[Part] | None = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """The arm at ``q`` as world-space polygons plus one shaded RGB per polygon."""
    parts = build_parts(arm) if parts is None else parts
    frames = arm.link_frames(q)
    polys: List[np.ndarray] = []
    colours: List[np.ndarray] = []
    for part in parts:
        rotation = frames[part.frame][:3, :3]
        origin = frames[part.frame][:3, 3]
        for poly in part.polys:
            world = poly @ rotation.T + origin
            polys.append(world)
            colours.append(shade(part.colour, _normal(world)))
    return polys, np.asarray(colours)


def swept_radius(arm: Puma560Arm) -> float:
    """Rough radius of the solid model about the waist axis, for framing plots."""
    return arm.a2 + arm.L3 + arm.tool + 0.06


# ----------------------------------------------------------------------
# the collision model
# ----------------------------------------------------------------------
#
# The evaluator has to answer "is any part of the robot inside a machine?", and
# it has to answer it tens of thousands of times.  Testing the 164 polygons
# against six boxes each time is far too slow, so the arm is bounded instead by
# a chain of spheres strung along its three moving link axes -- the upper arm,
# the forearm and the tool.
#
# The radii are *measured off the drawn solid*, not chosen: every point of the
# mesh in a link's frame is assigned to the nearest sphere centre on that
# link's axis, and the radius is the furthest such point.  The union of spheres
# therefore contains the model that gets drawn, which is the property that
# matters -- a layout the evaluator passes cannot show the arm inside a box in
# any of the figures.  :mod:`arm_study.selftest` checks the containment
# directly.
#
# The base and the waist are not in the chain, and do not need to be: the
# pedestal reaches 0.175 m from the waist axis and the cell already keeps
# machines outside a 0.18 m keep-out disc, while the turret and shoulder
# casting reach further (0.256 m) but only above z = 0.517 m, which is well
# clear of the tallest machine in the family.  `selftest` checks both.

#: How many spheres bound each link.  More is tighter and slower; these are
#: the smallest counts at which the bound stops being visibly baggy.
SPHERES_PER_LINK: Tuple[int, int, int] = (14, 14, 5)


@dataclass(frozen=True)
class CollisionSpheres:
    """The arm's bounding spheres, ready for :meth:`Puma560Arm.points_along_arm`.

    ``coefficients`` is ``(n, 3)`` -- distance along the upper arm, the forearm
    and the tool -- so one vectorised call places every centre in the world.
    ``radii`` is ``(n,)``.

    ``link`` says which link each sphere bounds (0 upper arm, 1 forearm,
    2 tool) and is carried explicitly rather than inferred, because the
    coefficients cannot tell: the elbow is both the far end of the upper arm
    and the near end of the forearm, so ``(a2, 0, 0)`` names two spheres with
    two different radii.  :attr:`is_tool` marks the wrist-and-gripper spheres,
    which the evaluator exempts while the tool is descending onto the very
    machine it is serving.
    """

    coefficients: np.ndarray
    radii: np.ndarray
    link: np.ndarray

    @property
    def is_tool(self) -> np.ndarray:
        return self.link == 2


def _densify(polys: Sequence[np.ndarray], step: float = 0.006) -> np.ndarray:
    """Points along every polygon edge, spaced at most ``step`` apart.

    Sweeps and boxes have vertices only at their ends, so bucketing raw
    vertices would leave the middle of a link bounded by nothing.  The
    perpendicular extent of these shapes varies linearly along an edge and the
    extent over a flat polygon is attained on its boundary, so walking the
    edges captures the surface exactly.
    """
    out: List[np.ndarray] = []
    for poly in polys:
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            n = max(2, int(np.linalg.norm(b - a) / step) + 2)
            out.append(a + np.linspace(0.0, 1.0, n)[:, None] * (b - a))
    return np.concatenate(out)


def collision_spheres(
    arm: Puma560Arm, counts: Tuple[int, int, int] = SPHERES_PER_LINK
) -> CollisionSpheres:
    """Bounding spheres for the three moving links, measured off the mesh."""
    parts = build_parts(arm)
    # (frame, link length, which of the three coefficient slots it fills)
    links = ((2, arm.a2, 0), (3, arm.L3, 1), (4, arm.tool, 2))

    coefficients: List[np.ndarray] = []
    radii: List[np.ndarray] = []
    link: List[np.ndarray] = []
    for (frame, length, slot), n in zip(links, counts):
        surface = _densify([poly for part in parts if part.frame == frame
                            for poly in part.polys])
        arc = np.linspace(0.0, length, n)
        nearest = np.argmin(np.abs(surface[:, 0][:, None] - arc[None, :]), axis=1)
        radius = np.zeros(n)
        for i in range(n):
            owned = surface[nearest == i]
            if owned.size:
                radius[i] = float(
                    np.linalg.norm(owned - np.array([arc[i], 0.0, 0.0]), axis=1).max()
                )
        block = np.zeros((n, 3))
        block[:, slot] = arc
        block[:, :slot] = (arm.a2, arm.L3)[:slot]
        # A centre that owns no surface -- the forearm casting stops short of
        # the wrist, so the last one on that link does -- bounds nothing, and
        # its points are already owned by a neighbour.  Dropping it keeps the
        # chain honest about what it covers.
        keep = radius > 0.0
        coefficients.append(block[keep])
        radii.append(radius[keep])
        link.append(np.full(int(keep.sum()), slot, dtype=int))

    return CollisionSpheres(
        coefficients=np.concatenate(coefficients),
        radii=np.concatenate(radii),
        link=np.concatenate(link),
    )


def mesh_surface_points(arm: Puma560Arm, frame: int, step: float = 0.006) -> np.ndarray:
    """Dense points on one link's drawn surface, in that link's frame.

    Only used by the self-test, which checks that the spheres really do
    contain the solid they claim to bound.
    """
    return _densify([poly for part in build_parts(arm) if part.frame == frame
                     for poly in part.polys], step)


# ----------------------------------------------------------------------
# the browser's copy
# ----------------------------------------------------------------------


def viewer_payload(arm: Puma560Arm, decimals: int = 4) -> Dict:
    """The solid model in a form :mod:`arm_study.viewer` can rebuild in JS.

    Local polygons and colours travel once; the page applies the link
    transforms itself from the four joint angles of whichever frame it is
    showing.  Sending baked world geometry per animation frame would be two
    orders of magnitude larger and could not re-shade as the camera moves.
    """
    parts = build_parts(arm)
    return {
        "geom": {
            "d1": arm.d1, "d3": arm.d3, "a2": arm.a2, "L3": arm.L3, "tool": arm.tool,
            "base": [float(arm.base[0]), float(arm.base[1])],
        },
        "light": [float(v) for v in LIGHT],
        "ambient": AMBIENT,
        "parts": [
            {
                "f": part.frame,
                "c": [int(round(255 * c)) for c in part.colour],
                "p": [np.round(poly, decimals).tolist() for poly in part.polys],
            }
            for part in parts
        ],
    }
