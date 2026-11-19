"""Label a transfer-cube episode from simulator state, without watching it (issue B3).

The task: the right gripper picks the cube up and hands it to the left gripper. The simulator's
reward stage alone cannot separate the failures, because stage 3 ("left gripper touches the
cube") is also reached when the left arm bumps a cube that is lying on the table. The labels
below therefore use the contact flags and the cube height directly.
"""

from __future__ import annotations

from collections.abc import Sequence

SUCCESS = "success"
NEVER_GRASPED = "never_grasped"
GRASPED_THEN_DROPPED = "grasped_then_dropped"
LIFTED_NO_HANDOVER = "lifted_no_handover"
HANDOVER_FAILED = "handover_failed"

FAILURE_CATEGORIES = (NEVER_GRASPED, GRASPED_THEN_DROPPED, LIFTED_NO_HANDOVER, HANDOVER_FAILED)

DEFINITIONS = {
    NEVER_GRASPED: "The right gripper never held the cube clear of the table.",
    GRASPED_THEN_DROPPED: "The right gripper lifted the cube, then lost it before the left gripper touched it.",
    LIFTED_NO_HANDOVER: "The right gripper still held the cube in the air at the end; the left gripper never touched it.",
    HANDOVER_FAILED: "The left gripper touched the lifted cube, but never held it clear of the table.",
}

CUBE_REST_Z = 0.02
LIFT_CLEARANCE = 0.01  # metres above rest before the cube counts as lifted
MIN_LIFT_STEPS = 3  # consecutive steps, so a bounce or a nudge is not a lift


def classify(
    stage: Sequence[int],
    right_touch: Sequence[bool],
    left_touch: Sequence[bool],
    table_touch: Sequence[bool],
    cube_z: Sequence[float],
) -> dict:
    """Return the label and the events it was derived from. All inputs hold one value per step."""
    n = len(stage)
    held = [
        right_touch[t] and not table_touch[t] and cube_z[t] > CUBE_REST_Z + LIFT_CLEARANCE for t in range(n)
    ]
    first_lift = None
    run = 0
    for t in range(n):
        run = run + 1 if held[t] else 0
        if run == MIN_LIFT_STEPS:
            first_lift = t - MIN_LIFT_STEPS + 1
            break

    left_after_lift = None
    if first_lift is not None:
        left_after_lift = next((t for t in range(first_lift, n) if left_touch[t]), None)

    events = {
        "first_right_touch_step": next((t for t in range(n) if right_touch[t]), None),
        "first_lift_step": first_lift,
        "first_left_touch_after_lift_step": left_after_lift,
        "max_cube_z": max(cube_z) if n else None,
        "final_cube_z": cube_z[-1] if n else None,
        "held_by_right_at_end": bool(held[-1]) if n else False,
    }

    if n and max(stage) == 4:
        label = SUCCESS
    elif first_lift is None:
        label = NEVER_GRASPED
    elif left_after_lift is not None:
        label = HANDOVER_FAILED
    elif held[-1]:
        label = LIFTED_NO_HANDOVER
    else:
        label = GRASPED_THEN_DROPPED
    return {"label": label, **events}
