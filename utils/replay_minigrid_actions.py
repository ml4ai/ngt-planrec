#!/usr/bin/env python3
"""
Replay extracted actions in a MiniGrid environment for visualization.

Notes:
- This uses a simple empty grid (no obstacles) and replays relative moves
  from the action codes (0–7 for 8 directions). Non-move actions keep position.
- If x/z coordinates are present, we align the initial position and then
  follow movement deltas derived from consecutive coordinates; otherwise we
  rely solely on action codes for movement.
- This is a visualization aid; it does not simulate Saturn maps or obstacles.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gym

# Local MiniGrid modules
from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.minigrid import MiniGridEnv

# Our action code definitions (mirror new_study_action_extractor)
MOVE_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),   # left
    1: (1, 0),    # right
    2: (0, -1),   # up
    3: (0, 1),    # down
    4: (-1, -1),  # up-left
    5: (1, -1),   # up-right
    6: (-1, 1),   # down-left
    7: (1, 1),    # down-right
}
NO_ACTION = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay extracted actions in MiniGrid")
    parser.add_argument(
        "--trial-dir",
        type=Path,
        required=True,
        help="Directory containing actions_*.json (e.g., data/new_study_actions/T000602_Saturn_B)",
    )
    parser.add_argument(
        "--role",
        type=str,
        choices=["engineer", "medic", "transporter"],
        required=True,
        help="Role to replay",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Playback speed (frames per second)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=31,
        help="Size of the empty grid (walls on boundary)",
    )
    parser.add_argument(
        "--use_coords",
        action="store_true",
        help="Use x/z coordinate deltas if available to drive movement",
    )
    return parser.parse_args()


def load_actions_file(trial_dir: Path, role: str) -> Dict[str, List]:
    path = trial_dir / f"actions_{role}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing actions file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def compute_moves_from_actions(actions: List[int]) -> List[Tuple[int, int]]:
    deltas: List[Tuple[int, int]] = []
    for a in actions:
        if a in MOVE_DELTAS:
            deltas.append(MOVE_DELTAS[a])
        else:
            deltas.append((0, 0))
    return deltas


def compute_moves_from_coords(xs: List[Optional[float]], zs: List[Optional[float]]) -> List[Tuple[int, int]]:
    deltas: List[Tuple[int, int]] = []
    prev: Optional[Tuple[int, int]] = None
    for x, z in zip(xs, zs):
        if x is None or z is None:
            deltas.append((0, 0))
            prev = None
            continue
        pos = (int(round(x)), int(round(z)))
        if prev is None:
            deltas.append((0, 0))
        else:
            dx = max(-1, min(1, pos[0] - prev[0]))
            dz = max(-1, min(1, pos[1] - prev[1]))
            deltas.append((dx, dz))
        prev = pos
    return deltas


def rotate_agent_to(env: MiniGridEnv, target_dir: int) -> None:
    # env.agent_dir in {0: right, 1: down, 2: left, 3: up}
    while env.agent_dir != target_dir:
        # Rotate shortest direction
        diff = (target_dir - env.agent_dir) % 4
        if diff == 1:
            env.step(env.actions.right)
        elif diff == 3:
            env.step(env.actions.left)
        else:
            env.step(env.actions.right)


def step_absolute_delta(env: MiniGridEnv, dx: int, dz: int) -> None:
    # Map absolute grid delta to rotations + forward moves
    # Diagonals are decomposed into two axial moves
    steps: List[Tuple[int, int]] = []
    if dx != 0:
        steps.append((dx, 0))
    if dz != 0:
        steps.append((0, dz))
    if not steps:
        return
    for sdx, sdz in steps:
        if sdx == 1:
            rotate_agent_to(env, 0)  # right
        elif sdx == -1:
            rotate_agent_to(env, 2)  # left
        elif sdz == 1:
            rotate_agent_to(env, 1)  # down
        elif sdz == -1:
            rotate_agent_to(env, 3)  # up
        env.step(env.actions.forward)


def main() -> None:
    args = parse_args()
    data = load_actions_file(args.trial_dir, args.role)
    actions: List[int] = data.get("actions", [])
    xs: List[Optional[float]] = data.get("x_coordinates", [])
    zs: List[Optional[float]] = data.get("z_coordinates", [])

    # Initialize a simple empty MiniGrid environment
    env: MiniGridEnv = EmptyEnv(size=args.grid_size)
    env.reset()

    # Center start to have room around
    # Note: grid coordinates are (x, y); we use y as "z" here
    cx = env.width // 2
    cy = env.height // 2
    env.agent_pos = (cx, cy)
    env.agent_dir = 0  # face right initially

    # Derive movement deltas
    if args.use_coords and xs and zs and len(xs) == len(actions) and len(zs) == len(actions):
        deltas = compute_moves_from_coords(xs, zs)
    else:
        deltas = compute_moves_from_actions(actions)

    # Replay
    delay = 1.0 / max(1e-6, args.fps)
    for i, (dx, dz) in enumerate(deltas):
        # Non-move actions are ignored for visualization; you may print or overlay labels if desired
        if dx != 0 or dz != 0:
            step_absolute_delta(env, dx, dz)
        env.render(mode="human")
        time.sleep(delay)

    # Keep window for a short time at the end
    time.sleep(1.0)
    env.close()


if __name__ == "__main__":
    main()


