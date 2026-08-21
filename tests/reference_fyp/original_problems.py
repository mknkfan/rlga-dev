"""The instance generator used in the originally submitted study.
"""

import random

import numpy as np

from core.datastruct import Machine, Point


def generate_problem(problem_idx: int):
    """
    Generate one problem instance with 8 machines.
    We vary machine sizes, shapes and access points per problem.
    """
    # For reproducibility of problem definitions
    base_seed = 1000 + problem_idx
    random.seed(base_seed)
    np.random.seed(base_seed)

    workspace_bounds = (-15, 15, -15, 15)
    robot_position = Point(0, 0)

    machines = []
    for i in range(8):
        m_id = i + 1

        # Alternate shapes for variety
        shape = "l_shape" if i % 2 == 0 else "rectangle"

        # Random-ish but reasonable machine sizes
        width = random.uniform(3.0, 6.0)
        height = random.uniform(2.5, 5.0)

        # Access point somewhere near machine center
        ap_x = random.uniform(-1.5, 1.5)
        ap_y = random.uniform(-1.5, 1.5)

        if shape == "l_shape":
            # L-cutout smaller than total width/height
            l_cutout_width = random.uniform(1.0, width / 2.0)
            l_cutout_height = random.uniform(1.0, height / 2.0)
            machine = Machine(
                id=m_id,
                shape="l_shape",
                width=width,
                height=height,
                access_point=Point(ap_x, ap_y),
                l_cutout_width=l_cutout_width,
                l_cutout_height=l_cutout_height,
            )
        else:
            machine = Machine(
                id=m_id,
                shape="rectangle",
                width=width,
                height=height,
                access_point=Point(ap_x, ap_y),
            )

        machines.append(machine)

    # Simple sequence: visit machines 1..8
    sequence = list(range(1, 9))

    return machines, sequence, robot_position, workspace_bounds
