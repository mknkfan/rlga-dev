"""
Assert that the ``test`` split of ``core.problems`` is bit-identical to
the instance generator used in the original submission
(``tests.reference_fyp.original_problems.generate_problem``), so the revised numbers are directly
comparable with the previously reported ones.

Run:  python -m tests.test_problem_parity
"""

from __future__ import annotations

import sys

from core.problems import build_instances
from tests.reference_fyp.original_problems import generate_problem


def main() -> int:
    instances = build_instances("test")
    if len(instances) < 10:
        print(f"FAIL: expected at least 10 test instances, got {len(instances)}")
        return 1

    # The split may have been enlarged via FYP_TEST_INSTANCES; the originally
    # submitted instances are the first 10 and must stay bit-identical.
    for k, instance in enumerate(instances[:10]):
        machines, sequence, robot_position, workspace_bounds = generate_problem(k)

        if sequence != instance.sequence:
            print(f"FAIL [{instance.name}]: sequence {instance.sequence} != {sequence}")
            return 1
        if tuple(workspace_bounds) != tuple(instance.workspace_bounds):
            print(f"FAIL [{instance.name}]: workspace bounds differ")
            return 1
        if (robot_position.x, robot_position.y) != (
            instance.robot_position.x,
            instance.robot_position.y,
        ):
            print(f"FAIL [{instance.name}]: robot position differs")
            return 1
        if len(machines) != len(instance.machines):
            print(f"FAIL [{instance.name}]: machine count differs")
            return 1

        for expected, actual in zip(machines, instance.machines):
            fields = (
                ("id", expected.id, actual.id),
                ("shape", expected.shape, actual.shape),
                ("width", expected.width, actual.width),
                ("height", expected.height, actual.height),
                ("access_x", expected.access_point.x, actual.access_point.x),
                ("access_y", expected.access_point.y, actual.access_point.y),
                ("cutout_w", expected.l_cutout_width, actual.l_cutout_width),
                ("cutout_h", expected.l_cutout_height, actual.l_cutout_height),
            )
            for field_name, want, got in fields:
                if want != got:
                    print(
                        f"FAIL [{instance.name}] machine {expected.id}: "
                        f"{field_name} {got!r} != {want!r}"
                    )
                    return 1

    checked = min(10, len(instances))
    print(
        f"OK: the first {checked} of {len(instances)} test instances match "
        "tests.reference_fyp.original_problems.generate_problem"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
