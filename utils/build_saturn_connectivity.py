#!/usr/bin/env python3
"""
Build directional connectivity graphs for the Saturn facility map.

The script combines geometric bounds from ``Saturn_2.6_3D_sm_v1.0.json`` with the
high-level adjacency specification in ``Saturn_adjacency_by_level.json`` to
determine, for every connected pair of regions, whether the neighbor lies north,
south, east, west, or overlaps the source region. It saves the enriched graph as
JSON and emits Matplotlib visualizations to help verify the spatial wiring.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


LOGGER = logging.getLogger(__name__)

Direction = Literal["north", "south", "east", "west", "overlap"]
ORIENTATIONS: Tuple[Direction, ...] = ("north", "south", "east", "west", "overlap")
ARROW_COLORS: Mapping[Direction, str] = {
    "north": "#1f77b4",
    "south": "#d62728",
    "east": "#2ca02c",
    "west": "#ff7f0e",
    "overlap": "#7f7f7f",
}


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned rectangle representing a map location."""

    min_x: float
    max_x: float
    min_z: float
    max_z: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_z - self.min_z

    @property
    def center(self) -> Tuple[float, float]:
        return (self.min_x + self.width / 2.0, self.min_z + self.height / 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        type=Path,
        default=Path("data/map_excel/Saturn_2.6_3D_sm_v1.0.json"),
        help="Path to the Saturn map geometry JSON.",
    )
    parser.add_argument(
        "--adjacency",
        type=Path,
        default=Path("data/Saturn_adjacency_by_level.json"),
        help="Path to the level-based adjacency specification.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/saturn_directional_connectivity.json"),
        help="Destination JSON for the enriched directional graphs.",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=Path("gym_minigrid/envs/resources/vis"),
        help="Directory that will receive connectivity visualizations.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity for diagnostic logging.",
    )
    return parser.parse_args()


def load_location_bounds(map_path: Path) -> Dict[str, BoundingBox]:
    data = json.loads(map_path.read_text())
    locations: List[Mapping[str, object]] = data.get("locations", [])
    bounds: Dict[str, BoundingBox] = {}
    pending: Dict[str, Mapping[str, object]] = {}

    for location in locations:
        loc_id = location.get("id")
        if not loc_id:
            continue
        raw_bounds = location.get("bounds")
        if raw_bounds:
            bbox = parse_bounds(raw_bounds)
            if bbox:
                bounds[loc_id] = bbox
            continue
        pending[loc_id] = location

    changed = True
    while changed and pending:
        changed = False
        for loc_id, location in list(pending.items()):
            child_ids = location.get("child_locations")
            if not isinstance(child_ids, list) or not child_ids:
                continue
            child_boxes: List[BoundingBox] = []
            for child_id in child_ids:
                child_box = bounds.get(child_id)
                if child_box is None:
                    break
                child_boxes.append(child_box)
            else:
                bounds[loc_id] = union_boxes(child_boxes)
                pending.pop(loc_id)
                changed = True

    if pending:
        LOGGER.debug("Could not derive bounds for %d locations", len(pending))

    LOGGER.info("Loaded bounds for %d locations", len(bounds))
    return bounds


def parse_bounds(bounds: Mapping[str, object]) -> Optional[BoundingBox]:
    coord_entries = bounds.get("coordinates")
    if not isinstance(coord_entries, Sequence):
        return None

    xs: List[float] = []
    zs: List[float] = []
    for entry in coord_entries:
        if not isinstance(entry, Mapping):
            continue
        x = entry.get("x")
        z = entry.get("z")
        if x is None or z is None:
            continue
        xs.append(float(x))
        zs.append(float(z))

    if not xs or not zs:
        return None

    min_x = min(xs)
    max_x = max(xs)
    min_z = min(zs)
    max_z = max(zs)
    if min_x == max_x:
        max_x += 0.5  # avoid zero-sized extents for plotting
    if min_z == max_z:
        max_z += 0.5
    return BoundingBox(min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z)


def union_boxes(boxes: Sequence[BoundingBox]) -> BoundingBox:
    min_x = min(box.min_x for box in boxes)
    max_x = max(box.max_x for box in boxes)
    min_z = min(box.min_z for box in boxes)
    max_z = max(box.max_z for box in boxes)
    return BoundingBox(min_x=min_x, max_x=max_x, min_z=min_z, max_z=max_z)


def load_adjacency(adjacency_path: Path) -> Dict[str, Dict[str, List[str]]]:
    data = json.loads(adjacency_path.read_text())
    adjacency: Dict[str, Dict[str, List[str]]] = {}
    for level, mapping in data.items():
        if not isinstance(mapping, Mapping):
            continue
        adjacency[level] = {
            node: list(neighbors)
            for node, neighbors in mapping.items()
            if isinstance(neighbors, list)
        }
    LOGGER.info("Loaded adjacency for %d levels", len(adjacency))
    return adjacency


def ranges_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return not (a_max < b_min or b_max < a_min)


def infer_direction(src: BoundingBox, dst: BoundingBox) -> Direction:
    same_x = ranges_overlap(src.min_x, src.max_x, dst.min_x, dst.max_x)
    same_z = ranges_overlap(src.min_z, src.max_z, dst.min_z, dst.max_z)

    if dst.min_z >= src.max_z and same_x:
        return "south"
    if dst.max_z <= src.min_z and same_x:
        return "north"
    if dst.min_x >= src.max_x and same_z:
        return "east"
    if dst.max_x <= src.min_x and same_z:
        return "west"

    src_cx, src_cz = src.center
    dst_cx, dst_cz = dst.center
    dx = dst_cx - src_cx
    dz = dst_cz - src_cz

    if abs(dx) > abs(dz):
        return "east" if dx > 0 else "west"
    if abs(dz) > abs(dx):
        return "south" if dz > 0 else "north"
    return "overlap"


def build_directional_graph(
    adjacency: Mapping[str, Mapping[str, Sequence[str]]],
    bounds: Mapping[str, BoundingBox],
) -> Dict[str, Dict[str, List[str]]]:
    graph: Dict[str, Dict[str, List[str]]] = {}
    for node, neighbors in adjacency.items():
        src_bounds = bounds.get(node)
        if src_bounds is None:
            LOGGER.warning("Missing bounds for %s, skipping", node)
            continue
        orientation_map: MutableMapping[Direction, List[str]] = {d: [] for d in ORIENTATIONS}
        for neighbor in neighbors:
            dst_bounds = bounds.get(neighbor)
            if dst_bounds is None:
                LOGGER.warning("Missing bounds for neighbor %s (from %s)", neighbor, node)
                continue
            direction = infer_direction(src_bounds, dst_bounds)
            orientation_map[direction].append(neighbor)
        graph[node] = {
            direction: sorted(set(targets))
            for direction, targets in orientation_map.items()
            if targets
        }
    return graph


def save_graph(
    output_path: Path,
    level_graphs: Mapping[str, Dict[str, Dict[str, List[str]]]],
    map_path: Path,
    adjacency_path: Path,
) -> None:
    payload = {
        "levels": level_graphs,
        "metadata": {
            "source_map": str(map_path),
            "source_adjacency": str(adjacency_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    LOGGER.info("Wrote connectivity JSON to %s", output_path)


def plot_level(
    level: str,
    graph: Mapping[str, Mapping[Direction, Sequence[str]]],
    bounds: Mapping[str, BoundingBox],
    output_dir: Path,
) -> None:
    level_bounds = {node: bounds[node] for node in graph.keys() if node in bounds}
    if not level_bounds:
        LOGGER.warning("No drawable bounds found for level %s", level)
        return

    min_x = min(b.min_x for b in level_bounds.values())
    max_x = max(b.max_x for b in level_bounds.values())
    min_z = min(b.min_z for b in level_bounds.values())
    max_z = max(b.max_z for b in level_bounds.values())
    margin = 2.0

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Saturn Connectivity Level {level}")

    for node, bbox in level_bounds.items():
        rect = Rectangle(
            (bbox.min_x, bbox.min_z),
            bbox.width,
            bbox.height,
            fill=False,
            edgecolor="#cccccc",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        cx, cz = bbox.center
        ax.text(cx, cz, node, fontsize=4, ha="center", va="center")

    drawn_edges: set[Tuple[str, str]] = set()
    for src, orientation_map in graph.items():
        src_bbox = bounds[src]
        src_cx, src_cz = src_bbox.center
        for direction, neighbors in orientation_map.items():
            color = ARROW_COLORS.get(direction, "#000000")
            for neighbor in neighbors:
                key = tuple(sorted((src, neighbor)))
                if key in drawn_edges:
                    continue
                dst_bbox = bounds.get(neighbor)
                if dst_bbox is None:
                    continue
                dst_cx, dst_cz = dst_bbox.center
                ax.annotate(
                    "",
                    xy=(dst_cx, dst_cz),
                    xytext=(src_cx, src_cz),
                    arrowprops=dict(arrowstyle="->", color=color, linewidth=0.6, alpha=0.7),
                )
                drawn_edges.add(key)

    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_z - margin, max_z + margin)
    ax.invert_yaxis()  # align with top-left origin convention
    ax.grid(True, linewidth=0.2, color="#f0f0f0")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"saturn_connectivity_level_{level}.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved visualization for level %s to %s", level, figure_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    bounds = load_location_bounds(args.map)
    adjacency = load_adjacency(args.adjacency)

    level_graphs: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for level, level_adj in adjacency.items():
        graph = build_directional_graph(level_adj, bounds)
        level_graphs[level] = graph
        plot_level(level, graph, bounds, args.vis_dir)

    save_graph(args.output, level_graphs, args.map, args.adjacency)


if __name__ == "__main__":
    main()

