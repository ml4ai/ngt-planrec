"""Extract per-role action timelines for the new ASIST study.

This script ingests filtered metadata files (see ``filter_metadata.py``) and
produces structured action traces and victim histories for each trial in the
new multi-agent study.  The extraction logic is inspired by the legacy
``action_parser.py`` but adapted to the new event schema and action set.

Key features
============

* Only topics relevant to movement and interaction are processed (player
  rubble-collapses are intentionally ignored).
* Mission detection distinguishes among Saturn_A, Saturn_B, Saturn_C, and Saturn_D.
* Per-trial output folders contain:
  - One JSON file per role (medic, engineer, transporter) with action events.
  - A victim history JSON capturing initialization and state changes.
  - A trial summary JSON with metadata.
* Regions are resolved using the rectangle bounds defined in
  ``Saturn_2.6_3D_sm_v1.0.json``.
* Mission timer entries with ``"Mission Timer not initialized."`` are ignored.
* Sampling begins once the mission timer reaches 15:00, matching analyst
  requested warm-up removal.
* Movements are sampled at 0.1 s and expressed as eight-direction grid moves.

Usage example
-------------

.. code-block:: bash

    python utils/new_study_action_extractor.py \
        --metadata data/trials/HSRData_TrialMessages_..._filtered.metadata \
        --regions data/map_excel/Saturn_2.6_3D_sm_v1.0.json \
        --output data/new_study_actions

Multiple ``--metadata`` arguments and glob patterns are supported.  By default
the script accumulates results for every supplied metadata file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration                                                                     
# ---------------------------------------------------------------------------

ALLOWED_TOPICS = {
    "observations/state",
    "observations/events/player/victim_picked_up",
    "observations/events/player/victim_placed",
    "observations/events/player/triage",
    "observations/events/player/rubble_destroyed",
    "observations/events/player/location",
}

ROLE_TOKENS = {
    "RED": "medic",
    "BLUE": "engineer",
    "GREEN": "transporter",
}

# Callsign to role mapping (from trial start metadata)
CALLSIGN_TO_ROLE = {
    "RED": "medic",
    "GREEN": "transporter",
    "BLUE": "engineer",
}

# Human-readable ordering for output files (unknown roles kept internally only)
ROLE_ORDER = ["medic", "engineer", "transporter"]

# Discrete action space (movement occupies codes 0-7 for the eight compass directions)
MOVE_VECTORS = {
    (-1, 0): (0, "move left (0)"),
    (1, 0): (1, "move right (1)"),
    (0, -1): (2, "move up (2)"),
    (0, 1): (3, "move down (3)"),
    (-1, -1): (4, "move up-left (4)"),
    (1, -1): (5, "move up-right (5)"),
    (-1, 1): (6, "move down-left (6)"),
    (1, 1): (7, "move down-right (7)"),
}

MOVE_CODES = {delta: code for delta, (code, _) in MOVE_VECTORS.items()}
MOVE_LABELS = {code: label for _, (code, label) in MOVE_VECTORS.items()}

# Interaction codes: 0-7 are movement, 8 is explicit idling, >=9 are discrete events.
NO_ACTION_CODE = 8
TRIAGE_CODE = 9
CLEAR_RUBBLE_CODE = 10
PICKUP_CODE = 11
DROP_CODE = 12

ACTION_LABELS = {
    NO_ACTION_CODE: f"no action ({NO_ACTION_CODE})",
    TRIAGE_CODE: f"triage victim ({TRIAGE_CODE})",
    CLEAR_RUBBLE_CODE: f"clear rubble ({CLEAR_RUBBLE_CODE})",
    PICKUP_CODE: f"pickup victim ({PICKUP_CODE})",
    DROP_CODE: f"drop victim ({DROP_CODE})",
}

# Allowed missions for the new study (A/B/C/D variants are now supported)
ALLOWED_MISSIONS = {"Saturn_A", "Saturn_B", "Saturn_C", "Saturn_D"}

# Mission timer threshold (seconds) after which sampling begins
MISSION_TIMER_THRESHOLD_SECONDS = 15 * 60


def parse_mission_timer(value: Optional[str]) -> Optional[int]:
    """Convert a ``MM:SS`` mission timer string into total seconds (remaining)."""

    if not value or not isinstance(value, str):
        return None
    cleaned = value.strip().replace(" ", "")
    match = re.match(r"^(\d+):(\d+)$", cleaned)
    if not match:
        return None
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    total_seconds = minutes * 60 + seconds
    return total_seconds


def format_mission_timer(seconds: int) -> str:
    """Render a mission timer string ``MM : SS`` from total seconds."""

    seconds = max(0, int(seconds))
    minutes, remainder = divmod(seconds, 60)
    return f"{minutes:02d} : {remainder:02d}"


# ---------------------------------------------------------------------------
# Helper dataclasses                                                     
# ---------------------------------------------------------------------------

@dataclass
class ActionEvent:
    """Container for a single action event."""

    type: str
    timestamp: Optional[str]
    mission_timer: Optional[str]
    topic: Optional[str]
    position: Optional[Dict[str, float]] = None
    region: Optional[str] = None
    details: Dict[str, object] = field(default_factory=dict)
    code: Optional[int] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "type": self.type,
            "timestamp": self.timestamp,
            "mission_timer": self.mission_timer,
            "topic": self.topic,
        }
        if self.position is not None:
            payload["position"] = self.position
        if self.region is not None:
            payload["region"] = self.region
        if self.details:
            payload["details"] = self.details
        if self.code is not None:
            payload["code"] = self.code
        if self.label is not None:
            payload["label"] = self.label
        return payload


@dataclass
class VictimHistory:
    """Track status updates for a victim."""

    victim_id: str
    color: Optional[str] = None
    initial_info: Dict[str, object] = field(default_factory=dict)
    history: List[Dict[str, object]] = field(default_factory=list)

    def add_event(self, event: Dict[str, object]) -> None:
        if not self.initial_info:
            # Record initial snapshot lazily
            snapshot = {
                key: event.get(key)
                for key in (
                    "timestamp",
                    "mission_timer",
                    "status",
                    "position",
                    "region",
                )
            }
            self.initial_info = {k: v for k, v in snapshot.items() if v is not None}
        self.history.append(event)


@dataclass
class ActionSlot:
    code: int
    label: str
    timestamp: Optional[str]
    mission_timer: Optional[str]
    region: Optional[str]
    position: Optional[Dict[str, float]]
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class StateFrame:
    timestamp: str
    mission_timer: Optional[str]
    position: Optional[Dict[str, float]]
    region: Optional[str]
    participant_id: Optional[str]
    playername: Optional[str]
    slots: List[ActionSlot] = field(default_factory=list)


@dataclass
class PendingEvent:
    timestamp: Optional[str]
    mission_timer: Optional[str]
    code: int
    label: str
    region: Optional[str]
    position: Optional[Dict[str, float]]
    details: Dict[str, object]
    playername: Optional[str]
    participant_id: Optional[str]


def parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def format_iso_timestamp(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def shift_timestamp(value: Optional[str], seconds: float) -> Optional[str]:
    if value is None:
        return None
    base = parse_iso_timestamp(value)
    if base is None:
        return value
    shifted = base + timedelta(seconds=seconds)
    return format_iso_timestamp(shifted)


def make_action_slot(
    code: int,
    label: str,
    timestamp: Optional[str],
    mission_timer: Optional[str],
    region: Optional[str],
    position: Optional[Dict[str, float]],
    details: Optional[Dict[str, object]] = None,
) -> ActionSlot:
    return ActionSlot(
        code=code,
        label=label,
        timestamp=timestamp,
        mission_timer=mission_timer,
        region=region,
        position=position,
        details=details or {},
    )


def make_no_action_slot(
    base_timestamp: Optional[str],
    mission_timer: Optional[str],
    region: Optional[str],
    position: Optional[Dict[str, float]],
    details: Optional[Dict[str, object]] = None,
    offset_seconds: float = 0.0,
) -> ActionSlot:
    """Construct a canonical idling slot (code 8) optionally shifted in time."""
    ts = shift_timestamp(base_timestamp, offset_seconds) if offset_seconds else base_timestamp
    payload = {'info': 'no_action'}
    if details:
        payload.update(details)
    return make_action_slot(
        code=NO_ACTION_CODE,
        label=ACTION_LABELS[NO_ACTION_CODE],
        timestamp=ts,
        mission_timer=mission_timer,
        region=region,
        position=position,
        details=payload,
    )


def assign_movement_slots(prev_frame: StateFrame, current_frame: StateFrame, timeline: AgentTimeline) -> None:
    """Assign a single move/no-action slot to the current 0.2 s frame.

    The integer-rounded delta between successive retained states determines
    whether we emit one of the eight canonical move codes or fall back to a
    no-action slot when no motion (or an unexpected teleport) is observed.
    """
    prev_pos = prev_frame.position or {}
    curr_pos = current_frame.position or {}

    prev_x = prev_pos.get("x")
    prev_z = prev_pos.get("z")
    curr_x = curr_pos.get("x")
    curr_z = curr_pos.get("z")

    if prev_x is None or prev_z is None or curr_x is None or curr_z is None:
        # Fallback to a no-action slot if coordinates are unavailable; reuse prior spatial context when possible.
        base_details = {"playername": current_frame.playername, "participant_id": current_frame.participant_id}
        fallback_region = current_frame.region or prev_frame.region
        if fallback_region in (None, "unknown") and prev_frame.region not in (None, "unknown"):
            fallback_region = prev_frame.region
        fallback_position = current_frame.position or prev_frame.position
        current_frame.slots = [
            make_no_action_slot(current_frame.timestamp, current_frame.mission_timer, fallback_region, fallback_position, base_details)
        ]
        return

    prev_x_int = int(round(prev_x))
    prev_z_int = int(round(prev_z))
    curr_x_int = int(round(curr_x))
    curr_z_int = int(round(curr_z))

    last_known_region = prev_frame.region if prev_frame.region not in (None, "unknown") else timeline.last_region_by_player.get(current_frame.playername or "", "")

    base_details = {
        "playername": current_frame.playername,
        "participant_id": current_frame.participant_id,
        "last_region": last_known_region,
    }

    dx = curr_x_int - prev_x_int
    dz = curr_z_int - prev_z_int
    delta = (dx, dz)

    # Treat large jumps (|Δ| > 1) as idle; these typically indicate teleports or missing states.
    if abs(dx) > 1 or abs(dz) > 1:
        region = current_frame.region
        if region in (None, "unknown"):
            region = prev_frame.region
        if region in (None, "unknown"):
            region = base_details.get("last_region")
        position = current_frame.position or prev_frame.position
        current_frame.slots = [
            make_no_action_slot(
                current_frame.timestamp,
                current_frame.mission_timer,
                region,
                position,
                {**base_details, "delta": {"x": dx, "z": dz}},
            )
        ]
        return

    if delta == (0, 0):
        region = current_frame.region
        if region in (None, "unknown") and prev_frame.region not in (None, "unknown"):
            region = prev_frame.region
        position = current_frame.position or prev_frame.position
        current_frame.slots = [
            make_no_action_slot(
                current_frame.timestamp,
                current_frame.mission_timer,
                region,
                position,
                {**base_details, "delta": {"x": dx, "z": dz}},
            )
        ]
        return

    code = MOVE_CODES.get(delta)
    if code is None:
        region = current_frame.region
        if region in (None, "unknown") and prev_frame.region not in (None, "unknown"):
            region = prev_frame.region
        position = current_frame.position or prev_frame.position
        current_frame.slots = [
            make_no_action_slot(
                current_frame.timestamp,
                current_frame.mission_timer,
                region,
                position,
                {**base_details, "delta": {"x": dx, "z": dz}},
            )
        ]
        return

    label = MOVE_LABELS[code]
    region = current_frame.region
    if region in (None, "unknown") and prev_frame.region not in (None, "unknown"):
        region = prev_frame.region
    position = current_frame.position or prev_frame.position
    current_frame.slots = [
        make_action_slot(
            code=code,
            label=label,
            timestamp=current_frame.timestamp,
            mission_timer=current_frame.mission_timer,
            region=region,
            position=position,
            details={**base_details, "delta": {"x": dx, "z": dz}},
        )
    ]


def apply_pending_events(timeline: AgentTimeline) -> None:
    """Inject queued interaction events into the nearest future state frame.

    Each pending event overwrites the target frame's single slot so the action
    inherits the state's timestamp, region, and position, satisfying the
    alignment requirements for downstream modelling.
    """
    for player, events in timeline.pending_events_by_player.items():
        frames = timeline.state_frames_by_player.get(player, [])
        if not frames:
            continue
        idx = 0
        total = len(frames)
        for event in events:
            event_dt = parse_iso_timestamp(event.timestamp) if event.timestamp else None
            target = frames[-1]
            if event_dt is not None:
                while idx < total:
                    frame_dt = parse_iso_timestamp(frames[idx].timestamp)
                    if frame_dt is None or frame_dt >= event_dt:
                        break
                    idx += 1
                if idx < total:
                    target = frames[idx]
            base_details = event.details.copy() if event.details else {}
            if event.playername:
                base_details.setdefault("playername", event.playername)
            if event.participant_id:
                base_details.setdefault("participant_id", event.participant_id)

            # Use the state's coordinates/region so actions align with the resampled timeline.
            region = target.region
            if region in (None, "unknown"):
                if event.region not in (None, "unknown"):
                    region = event.region
                else:
                    region = timeline.last_region_by_player.get(player)
            position = target.position or event.position
            mission_timer = target.mission_timer

            action_slot = make_action_slot(
                code=event.code,
                label=event.label,
                timestamp=target.timestamp,
                mission_timer=mission_timer,
                region=region,
                position=position,
                details=base_details,
            )
            target.slots = [action_slot]


def flatten_timeline(timeline: AgentTimeline) -> List[Dict[str, object]]:
    """Produce a chronologically ordered action stream for the given role."""
    apply_pending_events(timeline)

    combined: List[Tuple[datetime, StateFrame, str]] = []
    for player, frames in timeline.state_frames_by_player.items():
        for frame in frames:
            dt = parse_iso_timestamp(frame.timestamp) or datetime.fromtimestamp(0, tz=timezone.utc)
            combined.append((dt, frame, player))

    combined.sort(key=lambda item: item[0])

    flattened: List[Dict[str, object]] = []
    for _, frame, player in combined:
        slots = frame.slots or []
        if not slots:
            base_details = {"playername": frame.playername, "participant_id": frame.participant_id}
            slots = [
                make_no_action_slot(frame.timestamp, frame.mission_timer, frame.region, frame.position, base_details)
            ]
        for slot in slots:
            flattened.append(
                {
                    "playername": player,
                    "participant_id": frame.participant_id,
                    "timestamp": slot.timestamp or frame.timestamp,
                    "mission_timer": slot.mission_timer or frame.mission_timer,
                    "code": slot.code,
                    "label": slot.label,
                    "region": slot.region or frame.region,
                    "position": slot.position or frame.position,
                    "details": slot.details,
                }
            )

    timeline.flattened_slots = flattened
    return flattened


@dataclass
class AgentTimeline:
    """Aggregate actions for a given role."""

    role: str
    playernames: set[str] = field(default_factory=set)
    last_region_by_player: Dict[str, str] = field(default_factory=dict)
    participant_ids: set[str] = field(default_factory=set)
    player_to_participants: Dict[str, set[str]] = field(default_factory=dict)

    last_timer_by_player: Dict[str, int] = field(default_factory=dict)
    state_frames_by_player: Dict[str, List[StateFrame]] = field(default_factory=dict)
    state_keep_toggle: Dict[str, bool] = field(default_factory=dict)
    pending_events_by_player: Dict[str, List[PendingEvent]] = field(default_factory=dict)
    flattened_slots: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class TrialContext:
    """State accumulated for a single trial."""

    trial_id: str
    trial_tag: str
    mission: Optional[str] = None
    mission_source: Optional[str] = None
    timelines: Dict[str, AgentTimeline] = field(default_factory=dict)
    victims: Dict[str, VictimHistory] = field(default_factory=dict)
    player_role_map: Dict[str, str] = field(default_factory=dict)
    participant_to_player: Dict[str, str] = field(default_factory=dict)
    participant_to_role: Dict[str, str] = field(default_factory=dict)

    def get_timeline(self, role: str) -> AgentTimeline:
        timeline = self.timelines.get(role)
        if timeline is None:
            timeline = AgentTimeline(role=role)
            self.timelines[role] = timeline
        return timeline

    def register_player_role(self, playername: Optional[str], role: str, participant_id: Optional[str] = None) -> None:
        if playername and role:
            if playername not in self.player_role_map:
                self.player_role_map[playername] = role
        if participant_id:
            pid = participant_id.strip()
            if pid:
                if playername:
                    self.participant_to_player.setdefault(pid, playername)
                if role:
                    self.participant_to_role.setdefault(pid, role)

    def register_participant(self, participant_id: str, playername: Optional[str], callsign: Optional[str]) -> None:
        pid = participant_id.strip()
        if not pid:
            return
        player = playername.strip() if isinstance(playername, str) else None
        role = None
        if isinstance(callsign, str):
            role = CALLSIGN_TO_ROLE.get(callsign.strip().upper())
        if player and role:
            self.player_role_map.setdefault(player, role)
        if player:
            self.participant_to_player.setdefault(pid, player)
        if role:
            self.participant_to_role.setdefault(pid, role)

    def upsert_victim(self, victim_id: str, color: Optional[str] = None) -> VictimHistory:
        victim = self.victims.get(victim_id)
        if victim is None:
            victim = VictimHistory(victim_id=victim_id, color=color)
            self.victims[victim_id] = victim
        elif color and not victim.color:
            victim.color = color
        return victim


# ---------------------------------------------------------------------------
# Region utilities                                                       
# ---------------------------------------------------------------------------

@dataclass
class Region:
    region_id: str
    x_min: float
    x_max: float
    z_min: float
    z_max: float

    def contains(self, x: float, z: float) -> bool:
        return self.x_min <= x <= self.x_max and self.z_min <= z <= self.z_max


def load_regions(map_path: Path) -> List[Region]:
    """Load rectangular regions from the Saturn semantic map JSON."""

    with map_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    raw_locations = data.get("locations")
    if not isinstance(raw_locations, list):
        raise ValueError(f"Unexpected map format: 'locations' list missing in {map_path}")

    regions: List[Region] = []
    for location in raw_locations:
        if not isinstance(location, dict):
            continue
        bounds = location.get("bounds")
        if not isinstance(bounds, dict):
            continue
        if bounds.get("type") != "rectangle":
            continue
        coords = bounds.get("coordinates")
        if not isinstance(coords, list) or len(coords) < 2:
            continue
        xs = [coord.get("x") for coord in coords if "x" in coord]
        zs = [coord.get("z") for coord in coords if "z" in coord]
        if not xs or not zs:
            continue
        x_min, x_max = min(xs), max(xs)
        z_min, z_max = min(zs), max(zs)
        region_id = location.get("id") or "unknown"
        regions.append(Region(region_id=region_id, x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max))
    return regions


def find_region(regions: Iterable[Region], x: Optional[float], z: Optional[float]) -> str:
    """Resolve a semantic region for a coordinate pair with integer fallbacks.

    Coordinates streamed from the telemetry frequently land on tile centers while
    the semantic map stores rectangle bounds aligned to tile edges.  When an
    exact match fails, probe the floored integer tile and the corresponding tile
    center to minimise ``unknown`` classifications.
    """

    if x is None or z is None:
        return "unknown"

    probes = [(x, z)]
    floored_x = math.floor(x)
    floored_z = math.floor(z)
    if (floored_x, floored_z) != (x, z):
        probes.append((floored_x + 0.5, floored_z + 0.5))

    for probe_x, probe_z in probes:
        for region in regions:
            if region.contains(probe_x, probe_z):
                return region.region_id

    return "unknown"


# ---------------------------------------------------------------------------
# Parsing helpers                                                        
# ---------------------------------------------------------------------------

TIMESTAMP_FIELDS = ("timestamp", "@timestamp")


def extract_timestamp(entry: Dict[str, object]) -> Optional[str]:
    for source in (entry.get("msg"), entry.get("header"), entry):
        if isinstance(source, dict):
            for key in TIMESTAMP_FIELDS:
                if key in source:
                    return source[key]
    return None


def extract_trial_id(entry: Dict[str, object], fallback: str) -> str:
    for container_key in ("msg", "header", entry):
        container = entry if container_key is entry else entry.get(container_key)
        if isinstance(container, dict):
            for key in ("trial_id", "trialId", "trialID", "trial"):
                value = container.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    return fallback


def parse_file_trial_id(filename: str) -> Optional[str]:
    match = re.search(r"Trial-(T\d+)", filename)
    if match:
        return match.group(1)
    return None


def extract_mission(entry: Dict[str, object]) -> Optional[Tuple[str, str]]:
    data = entry.get("data")
    if isinstance(data, dict):
        mission = (
            data.get("experiment_mission")
            or data.get("mission")
            or data.get("map")
        )
        if isinstance(mission, str):
            normalized = normalize_mission(mission)
            if normalized:
                return normalized, "data"
    msg = entry.get("msg")
    if isinstance(msg, dict):
        mission = (
            msg.get("experiment_mission")
            or msg.get("mission")
            or msg.get("map")
        )
        if isinstance(mission, str):
            normalized = normalize_mission(mission)
            if normalized:
                return normalized, "msg"
    return None


def normalize_mission(value: str) -> Optional[str]:
    """Normalize free-form mission strings to canonical Saturn tags."""

    cleaned = value.strip()
    if not cleaned:
        return None
    match = re.search(r"saturn[\s_\-]*([abcd])", cleaned, re.IGNORECASE)
    if match:
        return f"Saturn_{match.group(1).upper()}"
    return None


def get_topic_from_entry(entry: Dict[str, object]) -> Optional[str]:
    topic = entry.get("topic")
    if isinstance(topic, str) and topic:
        return topic
    for container_key in ("msg", "data"):
        container = entry.get(container_key)
        if isinstance(container, dict):
            topic = container.get("topic")
            if isinstance(topic, str) and topic:
                return topic
    return None


def determine_role(entry: Dict[str, object], playername: Optional[str]) -> str:
    data = entry.get("data")
    if isinstance(data, dict):
        role = data.get("role") or data.get("player_role") or data.get("playerRole")
        if isinstance(role, str) and role.strip():
            return role.lower()
    if playername:
        upper_name = playername.upper()
        for token, role in ROLE_TOKENS.items():
            if token in upper_name:
                return role
    return "unknown"


def sanitize_role(role: str) -> str:
    normalized = role.lower().strip()
    if normalized in ROLE_ORDER:
        return normalized
    return "unknown"


def get_player_name(entry: Dict[str, object]) -> Optional[str]:
    data = entry.get("data")
    if isinstance(data, dict):
        for key in ("playername", "player_name", "playerId", "player_id", "participant_id"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def ensure_position(data: Dict[str, object]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    def _cast(value: object) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value)
            except ValueError:
                return None
        return None

    x = _cast(data.get("x") or data.get("loc_x") or data.get("position_x"))
    y = _cast(data.get("y") or data.get("loc_y") or data.get("position_y"))
    z = _cast(data.get("z") or data.get("loc_z") or data.get("position_z"))
    return x, y, z


def ensure_victim_position(data: Dict[str, object]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    x = data.get("victim_x") or data.get("x")
    z = data.get("victim_z") or data.get("z")
    y = data.get("victim_y") or data.get("y")
    def _cast(value: object) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value)
            except ValueError:
                return None
        return None
    return _cast(x), _cast(y), _cast(z)


def prepare_position_dict(x: Optional[float], y: Optional[float], z: Optional[float]) -> Optional[Dict[str, float]]:
    if x is None and y is None and z is None:
        return None
    payload: Dict[str, float] = {}
    if x is not None:
        payload["x"] = x
    if y is not None:
        payload["y"] = y
    if z is not None:
        payload["z"] = z
    return payload or None


def resolve_victim_id(data: Dict[str, object]) -> Optional[str]:
    for key in ("victim_id", "victimId", "victimID", "victim"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (int, float)):
            return str(value)
    return None


def resolve_color(data: Dict[str, object]) -> Optional[str]:
    color = data.get("color") or data.get("victim_color")
    if isinstance(color, str) and color.strip():
        return color.strip()
    return None


# ---------------------------------------------------------------------------
# Trial metadata helpers                                                     
# ---------------------------------------------------------------------------

def process_trial_metadata(entry: Dict[str, object], trial: TrialContext) -> None:
    if entry.get("topic") != "trial":
        return
    data = entry.get("data")
    if not isinstance(data, dict):
        return
    client_info = data.get("client_info")
    if isinstance(client_info, list):
        for info in client_info:
            if not isinstance(info, dict):
                continue
            participant_id = info.get("participant_id") or info.get("unique_id")
            playername = info.get("playername") or info.get("callsign")
            callsign = info.get("callsign")
            if isinstance(participant_id, str):
                trial.register_participant(participant_id, playername, callsign)


# ---------------------------------------------------------------------------
# Player / role resolution                                                  
# ---------------------------------------------------------------------------

def resolve_player_info(entry: Dict[str, object], trial: TrialContext) -> Tuple[Optional[str], str, Optional[str]]:
    data = entry.get("data", {})

    participant_id = data.get("participant_id") or data.get("participantId") or data.get("player_id")
    if isinstance(participant_id, str):
        participant_id = participant_id.strip()
    else:
        participant_id = None

    playername = data.get("playername") or data.get("player_name")
    if isinstance(playername, str):
        playername = playername.strip()

    if not playername and participant_id:
        playername = trial.participant_to_player.get(participant_id)

    role_from_data = data.get("role") or data.get("player_role")
    role = sanitize_role(role_from_data) if isinstance(role_from_data, str) else "unknown"

    if role == "unknown" and participant_id:
        role = sanitize_role(trial.participant_to_role.get(participant_id, "unknown"))

    if role == "unknown" and playername:
        role = sanitize_role(determine_role_from_name(playername))

    trial.register_player_role(playername, role, participant_id)
    return playername, role, participant_id


def determine_role_from_name(playername: str) -> str:
    upper_name = playername.upper()
    for token, role in ROLE_TOKENS.items():
        if token in upper_name:
            return role
    return "unknown"


# ---------------------------------------------------------------------------
# Processing logic                                                      
# ---------------------------------------------------------------------------

def process_state_entry(entry: Dict[str, object], trial: TrialContext, regions: List[Region]) -> None:
    """Translate continuous state updates into grid-aligned move actions.

    Once the mission timer reaches 15 minutes, every 0.1 s state update contributes
    exactly one action slot: either a single eight-direction move (codes 0-7)
    derived from the previous retained state or the explicit idle code 8.
    """
    data = entry.get("data", {})
    if not isinstance(data, dict):
        return
    player, role, participant_id = resolve_player_info(entry, trial)
    if not player:
        return

    mission_timer_str = data.get("mission_timer")
    mission_seconds = parse_mission_timer(mission_timer_str)

    # Ignore early warm-up samples until the mission timer reaches 15:00 or lower.
    if mission_seconds is None or mission_seconds > MISSION_TIMER_THRESHOLD_SECONDS:
        return

    timeline = trial.get_timeline(role)
    timeline.playernames.add(player)
    if participant_id:
        timeline.participant_ids.add(participant_id)
        participants = timeline.player_to_participants.setdefault(player, set())
        participants.add(participant_id)

    timestamp = extract_timestamp(entry)

    x, y, z = ensure_position(data)
    region_id = find_region(regions, x, z)
    position = prepare_position_dict(x, y, z)

    previous_region = timeline.last_region_by_player.get(player)
    resolved_region = region_id
    if resolved_region in (None, "unknown") and previous_region not in (None, "unknown"):
        resolved_region = previous_region

    frame = StateFrame(
        timestamp=timestamp or "",
        mission_timer=mission_timer_str,
        position=position,
        region=resolved_region,
        participant_id=participant_id,
        playername=player,
    )

    base_details = {"playername": player, "participant_id": participant_id}
    frame.slots = [make_no_action_slot(timestamp, mission_timer_str, resolved_region, position, base_details)]

    frames = timeline.state_frames_by_player.setdefault(player, [])
    frames.append(frame)
    if resolved_region in (None, "unknown"):
        timeline.last_region_by_player[player] = previous_region or resolved_region or ""
    else:
        timeline.last_region_by_player[player] = resolved_region
    if mission_seconds is not None:
        timeline.last_timer_by_player[player] = mission_seconds

    if len(frames) == 1:
        return

    prev_frame = frames[-2]
    assign_movement_slots(prev_frame, frame, timeline)


def process_victim_event(entry: Dict[str, object], trial: TrialContext, regions: List[Region], action_type: str) -> None:
    """Capture discrete interaction events with victims (pickup / drop / triage).

    Events are queued once the mission timer crosses the 15-minute threshold so
    they can overwrite the next retained state frame, mirroring the alignment of
    other discrete actions with the resampled timeline.
    """
    data = entry.get("data", {})
    if not isinstance(data, dict):
        return

    player, role, participant_id = resolve_player_info(entry, trial)

    mission_timer_str = data.get("mission_timer")
    mission_seconds = parse_mission_timer(mission_timer_str)
    timestamp = extract_timestamp(entry)

    victim_id = resolve_victim_id(data)
    victim_color = resolve_color(data)
    if not victim_id:
        victim_id = "unknown"

    x, y, z = ensure_victim_position(data)
    region_id = find_region(regions, x, z)
    position = prepare_position_dict(x, y, z)

    event_details = {
        "playername": player,
        "participant_id": participant_id,
        "role": role,
        "victim_id": victim_id,
        "victim_color": victim_color,
        "triage_state": data.get("triage_state"),
        "outcome": data.get("victim_outcome"),
    }

    timeline = trial.get_timeline(role)
    if player:
        timeline.playernames.add(player)
    if participant_id:
        timeline.participant_ids.add(participant_id)
        participants = timeline.player_to_participants.setdefault(player or "unknown", set())
        participants.add(participant_id)

    base_details = {k: v for k, v in event_details.items() if v is not None}

    # Defer queueing until the mission countdown reaches 15:00.
    if mission_seconds is None or mission_seconds > MISSION_TIMER_THRESHOLD_SECONDS:
        return

    if action_type == "triage":
        event_code = TRIAGE_CODE
        event_label = ACTION_LABELS[TRIAGE_CODE]
    elif action_type == "victim_picked_up":
        event_code = PICKUP_CODE
        event_label = ACTION_LABELS[PICKUP_CODE]
    elif action_type == "victim_placed":
        event_code = DROP_CODE
        event_label = ACTION_LABELS[DROP_CODE]
    else:
        event_code = None
        event_label = None

    if event_code is not None and player:
        pending = PendingEvent(
            timestamp=timestamp,
            mission_timer=mission_timer_str,
            code=event_code,
            label=event_label,
            region=region_id,
            position=position,
            details={k: v for k, v in event_details.items() if v is not None},
            playername=player,
            participant_id=participant_id,
        )
        timeline.pending_events_by_player.setdefault(player, []).append(pending)

    victim = trial.upsert_victim(victim_id, victim_color)
    victim_event = {
        "timestamp": timestamp,
        "mission_timer": mission_timer_str,
        "status": action_type,
        "playername": player,
        "role": role,
        "position": position,
        "region": region_id,
        "details": base_details,
    }
    victim.add_event(victim_event)

    if mission_seconds is not None and player:
        timeline.last_timer_by_player[player] = mission_seconds


def process_location_event(entry: Dict[str, object], trial: TrialContext, regions: List[Region]) -> None:
    return


def process_rubble_event(entry: Dict[str, object], trial: TrialContext, regions: List[Region], action_type: str) -> None:
    """Record engineer rubble interactions as discrete clear-rubble actions.

    Only rubble events occurring after the 15-minute mark are considered to keep
    alignment with the state resampling window.
    """
    data = entry.get("data", {})
    if not isinstance(data, dict):
        return
    player, role, participant_id = resolve_player_info(entry, trial)

    mission_timer_str = data.get("mission_timer")
    mission_seconds = parse_mission_timer(mission_timer_str)
    timestamp = extract_timestamp(entry)

    # Honor the 15-minute start threshold for rubble-related actions as well.
    if mission_seconds is None or mission_seconds > MISSION_TIMER_THRESHOLD_SECONDS:
        return

    x, y, z = ensure_position(data)
    region_id = find_region(regions, x, z)

    timeline = trial.get_timeline(role)
    if player:
        timeline.playernames.add(player)
    if participant_id:
        timeline.participant_ids.add(participant_id)
        participants = timeline.player_to_participants.setdefault(player or "unknown", set())
        participants.add(participant_id)

    pending = PendingEvent(
        timestamp=timestamp,
        mission_timer=mission_timer_str,
        code=CLEAR_RUBBLE_CODE,
        label=ACTION_LABELS[CLEAR_RUBBLE_CODE],
        region=region_id,
        position=prepare_position_dict(x, y, z),
        details={
            "playername": player,
            "role": role,
            "collapse_cause": data.get("cause"),
        },
        playername=player,
        participant_id=participant_id,
    )
    if player:
        timeline.pending_events_by_player.setdefault(player, []).append(pending)

    if mission_seconds is not None and player:
        timeline.last_timer_by_player[player] = mission_seconds


TOPIC_HANDLERS = {
    "observations/state": process_state_entry,
    "observations/events/player/victim_picked_up": lambda entry, trial, regions: process_victim_event(entry, trial, regions, "victim_picked_up"),
    "observations/events/player/victim_placed": lambda entry, trial, regions: process_victim_event(entry, trial, regions, "victim_placed"),
    "observations/events/player/triage": lambda entry, trial, regions: process_victim_event(entry, trial, regions, "triage"),
    "observations/events/player/rubble_destroyed": lambda entry, trial, regions: process_rubble_event(entry, trial, regions, "rubble_cleared"),
    "observations/events/player/location": process_location_event,
    "trial": process_trial_metadata,
}


# ---------------------------------------------------------------------------
# Output utilities                                                      
# ---------------------------------------------------------------------------

def write_trial_outputs(trial: TrialContext, output_dir: Path) -> None:
    mission_tag = trial.mission or "Unknown"
    trial_dir = output_dir / f"{trial.trial_tag}_{mission_tag}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    summary_roles: Dict[str, List[Dict[str, object]]] = {}
    for role in ROLE_ORDER:
        timeline = trial.timelines.get(role)
        if not timeline:
            continue
        players: List[Dict[str, object]] = []
        for player in sorted(timeline.playernames):
            participants = sorted(timeline.player_to_participants.get(player, set()))
            players.append({
                "playername": player,
                "participant_ids": participants,
            })
        summary_roles[role] = players

    summary = {
        "trial_id": trial.trial_id,
        "mission": trial.mission,
        "mission_source": trial.mission_source,
        "roles": summary_roles,
        "victim_count": len(trial.victims),
    }
    (trial_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    role_slot_map: Dict[str, List[Dict[str, object]]] = {}
    for role in ROLE_ORDER:
        timeline = trial.timelines.get(role)
        if timeline is None:
            continue
        flattened = flatten_timeline(timeline)
        role_slot_map[role] = flattened

        role_file = trial_dir / f"actions_{role}.json"
        payload = {
            "trial_id": trial.trial_id,
            "mission": trial.mission,
            "role": role,
            "players": sorted(timeline.playernames),
            "participant_ids": sorted(timeline.participant_ids),
            "player_participants": {
                player: sorted(timeline.player_to_participants.get(player, set()))
                for player in sorted(timeline.playernames)
            },
            "actions": [slot["code"] for slot in flattened],
            "action_labels": [slot["label"] for slot in flattened],
            "mission_timers": [slot.get("mission_timer") for slot in flattened],
            "regions": [slot.get("region") for slot in flattened],
            "x_coordinates": [slot.get("position", {}).get("x") if slot.get("position") else None for slot in flattened],
            "z_coordinates": [slot.get("position", {}).get("z") if slot.get("position") else None for slot in flattened],
        }
        role_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Remove legacy unknown-role files that may remain from earlier runs
    unknown_file = trial_dir / "actions_unknown.json"
    if unknown_file.exists():
        unknown_file.unlink()

    # Victim timeline at 0.2 s resolution aligned with action sequence length
    base_role = None
    for role in ROLE_ORDER:
        if role in role_slot_map:
            base_role = role
            break

    slot_times: List[datetime] = []
    if base_role:
        for slot in role_slot_map[base_role]:
            ts = slot.get("timestamp")
            parsed = parse_iso_timestamp(ts)
            if parsed is not None:
                slot_times.append(parsed)

    slot_times.sort()

    # Prepare victim states
    victim_states: Dict[str, Dict[str, object]] = {}
    victim_last_region: Dict[str, Optional[str]] = {}
    victim_last_position: Dict[str, Optional[Dict[str, float]]] = {}
    victim_events: List[Tuple[datetime, str, Dict[str, object]]] = []
    for victim in trial.victims.values():
        initial = victim.initial_info
        position = initial.get("position") if initial else None
        region = initial.get("region") if initial else None
        triaged = False
        victim_states[victim.victim_id] = {
            "position": position,
            "region": region,
            "triaged": triaged,
        }
        if region not in (None, "unknown"):
            victim_last_region[victim.victim_id] = region
        if position is not None:
            victim_last_position[victim.victim_id] = position
        for event in victim.history:
            ts = event.get("timestamp")
            parsed = parse_iso_timestamp(ts)
            if parsed is None:
                continue
            victim_events.append((parsed, victim.victim_id, event))

    victim_events.sort(key=lambda item: item[0])
    event_idx = 0
    total_events = len(victim_events)

    victim_timeline: List[Dict[str, object]] = []
    for slot_dt in slot_times:
        while event_idx < total_events and victim_events[event_idx][0] <= slot_dt:
            _, vid, event = victim_events[event_idx]
            state = victim_states.setdefault(vid, {"position": None, "region": None, "triaged": False})
            status = event.get("status")
            details = event.get("details", {})
            if status == "triage":
                triage_state = (details.get("triage_state") or "").upper()
                if triage_state == "SUCCESSFUL":
                    state["triaged"] = True
                position = event.get("position")
                if position:
                    state["position"] = position
                    victim_last_position[vid] = position
                region = event.get("region")
                if region:
                    state["region"] = region
                    if region != "unknown":
                        victim_last_region[vid] = region
            elif status in {"victim_picked_up", "victim_placed", "victim_evacuated"}:
                position = event.get("position")
                if position:
                    state["position"] = position
                    victim_last_position[vid] = position
                region = event.get("region")
                if region:
                    state["region"] = region
                    if region != "unknown":
                        victim_last_region[vid] = region
            event_idx += 1

        snapshot = {
            "timestamp": format_iso_timestamp(slot_dt),
            "victims": [
                {
                    "victim_id": vid,
                    "position": (
                        state.get("position")
                        if state.get("position") is not None
                        else victim_last_position.get(vid)
                    ),
                    "region": (
                        state.get("region")
                        if state.get("region") not in (None, "unknown")
                        else victim_last_region.get(vid)
                    ),
                    "triaged": state.get("triaged", False),
                }
                for vid, state in sorted(victim_states.items())
            ],
        }
        victim_timeline.append(snapshot)

    victims_payload = {
        "trial_id": trial.trial_id,
        "mission": trial.mission,
        "timeline": victim_timeline,
    }
    (trial_dir / "victims.json").write_text(json.dumps(victims_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main driver                                                          
# ---------------------------------------------------------------------------

def collect_metadata_files(patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_file():
            files.append(path)
        else:
            matched = list(Path().glob(pattern))
            files.extend(p for p in matched if p.is_file())
    return sorted(set(files))


def parse_file_trial_id(filename: str) -> Optional[str]:
    match = re.search(r"Trial-(T\d+)", filename)
    if match:
        return match.group(1)
    return None


def should_keep_entry(entry: Dict[str, object]) -> bool:
    """
    Check if an entry should be kept (i.e., has an allowed topic and valid mission_timer).
    
    Args:
        entry: A JSON object (dictionary) from the metadata file
        
    Returns:
        True if the entry has an allowed topic and valid mission_timer, False otherwise
    """
    # First check: exclude entries with invalid mission_timer
    data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
    mission_timer_data = data.get("mission_timer")
    mission_timer_top = entry.get("mission_timer")
    if mission_timer_data == "Mission Timer not initialized." or mission_timer_top == "Mission Timer not initialized.":
        return False

    # Second check: only keep entries with allowed topics
    topic = get_topic_from_entry(entry)
    if topic not in ALLOWED_TOPICS:
        return False
    return True


def process_metadata_file(metadata_path: Path, regions: List[Region], output_dir: Path) -> None:
    file_trial_tag = parse_file_trial_id(metadata_path.name)
    with metadata_path.open("r", encoding="utf-8") as fp:
        fallback_trial = metadata_path.stem
        trials: Dict[str, TrialContext] = {}

        for line_num, raw_line in enumerate(fp, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON at {metadata_path}:{line_num}")
                continue

            trial_id = extract_trial_id(entry, fallback_trial)
            trial = trials.get(trial_id)
            if trial is None:
                trial = TrialContext(trial_id=trial_id, trial_tag=file_trial_tag or trial_id)
                trials[trial_id] = trial

            mission_info = extract_mission(entry)
            if mission_info and not trial.mission:
                mission_value, source = mission_info
                trial.mission = mission_value
                trial.mission_source = source

            process_trial_metadata(entry, trial)

            if not should_keep_entry(entry):
                continue

            topic = get_topic_from_entry(entry)
            handler = TOPIC_HANDLERS.get(topic)
            if handler:
                handler(entry, trial, regions)

        for trial in trials.values():
            if (trial.mission or "") not in ALLOWED_MISSIONS:
                continue
            write_trial_outputs(trial, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract per-role actions for the new ASIST study")
    parser.add_argument(
        "--metadata",
        nargs="+",
        default=["data/trials/HSRData_TrialMessages_Trial-*.metadata"],
        help="Metadata files or glob patterns to process",
    )
    parser.add_argument(
        "--regions",
        default="data/map_excel/Saturn_2.6_3D_sm_v1.0.json",
        help="Path to Saturn semantic map JSON",
    )
    parser.add_argument(
        "--output",
        default="data/new_study_actions",
        help="Directory where per-trial action files will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_files = collect_metadata_files(args.metadata)
    if not metadata_files:
        raise FileNotFoundError("No metadata files matched the provided patterns")

    regions = load_regions(Path(args.regions))
    if not regions:
        raise ValueError("No rectangular regions were loaded from the semantic map")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metadata_file in metadata_files:
        print(f"Processing metadata file: {metadata_file}")
        process_metadata_file(metadata_file, regions, output_dir)


if __name__ == "__main__":
    main()
