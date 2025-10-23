"""
Excel map parser for ASIST Saturn maps.

This module parses two maps contained in `data/map_excel/asist_map.xlsx` and
produces two numpy arrays for MiniGrid-compatible maps. The expected sheet
names are `SaturnA_2.3` and `SaturnB_2.3`.

Changes from the legacy parser:
- Switch to `openpyxl` to read modern .xlsx files (xlrd dropped xlsx support).
- Support for new symbols: A/B/C victims, X collapse plate/threat collapse,
  D falling rubble, P victim detection plate, R/RR/RRR rubble layers,
  T freezing threat, F objects.
- Grey and brown filled cells are considered walls.
- No doors are produced anymore.
- Map extents are larger, spanning coordinates from (-2226,-13) in the
  top-left to (-2087, 64) in the bottom-right. This results in a fixed grid
  of height 78 and width 140.

Output files are saved under `gym_minigrid/envs/resources/` with names:
- `SaturnA_2_3.npy`
- `SaturnB_2_3.npy`
- `raw_map_state_saturnA.npy`, `raw_map_state_saturnB.npy` (walkable==1, walls==4)

All functions are annotated with type hints and in-line comments describe
non-obvious implementation details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Any, List

import numpy as np
from openpyxl import load_workbook
from openpyxl.cell.cell import Cell
from openpyxl.utils import column_index_from_string
try:
    # Older/newer openpyxl versions expose COLOR_INDEX here; fallback to empty
    from openpyxl.styles.colors import COLOR_INDEX  # type: ignore
except Exception:  # pragma: no cover - best-effort compatibility
    COLOR_INDEX = {}  # type: ignore


# Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "map_excel"
RESOURCES_DIR = (Path(__file__).parent / "./envs/resources").resolve()

# Workbook and sheet configuration
WORKBOOK_PATH = DATA_DIR / "asist_map.xlsx"
SHEET_NAMES = ("SaturnA_2.3", "SaturnB_2.3")

# Coordinate system from Excel range C5:EK80 (1-based). We detect the
# content origin, but for exact control we set explicit offsets:
# C=3, row 5 → origin; EK is column 5*26+?; use openpyxl to compute.
EXCEL_RANGE_START = (5, 3)   # (row, col) = (5, 'C')
EXCEL_RANGE_END = (80, None) # end row; end col resolved per sheet
TOP_LEFT: Tuple[int, int] = (-2225, -11)
BOTTOM_RIGHT: Tuple[int, int] = (-2087, 64)

# Derived grid size (height x width)
GRID_HEIGHT: int = BOTTOM_RIGHT[1] - TOP_LEFT[1] + 1
GRID_WIDTH: int = BOTTOM_RIGHT[0] - TOP_LEFT[0] + 1


# MiniGrid integer IDs used in this repository (see gym_minigrid/index_mapping.py)
EMPTY = 1
WALL = 4            # Default wall
WALL_HEAVY = 30     # Heavy wall
WALL_LIGHT = 31     # Light grey rubble wall
LAVA = 9            # Hazard (red)
BOX = 255           # Generic object (lighter brown)
BOX_LIGHT_BLUE = 11 # Light-blue plate (P)
BOX_DARK_BLUE = 12  # Dark-blue object (D)
BOX_RED = 13        # Red object if needed (not used for X)
GOAL_A = 81         # Victim A (green)
GOAL_B = 82         # Victim B (light green)
GOAL_C = 83         # Victim C (yellow)


# Colors used in the Excel for wall identification via fill
# These are common RGB hex strings (uppercase) for grey/brown families.
GREY_HEXES: Iterable[str] = {
    "FF808080", "FF7F7F7F", "FFC0C0C0", "FFBFBFBF", "FFB0B0B0",
    "FF999999", "FF666666", "FF4D4D4D", "FFD9D9D9"
}
BROWN_HEXES: Iterable[str] = {
    "FFCBA986", "FFA8947D", "FF8B4513", "FFA0522D", "FFCD853F",
    "FF7B3F00", "FF5C4033", "FF6F4E37"
}

# Any non-white, non-empty fill should be considered as structure unless
# explicitly mapped by a token; use this to catch additional wall colors.
NON_WALL_EXCEPTIONS: Iterable[str] = {
    # pure white and none
    "FFFFFFFF", "FF000000", "00000000"
}

# Dynamic exceptions collected from the sheet for cells that should be empty
# even if they have a non-white fill. We track both ARGB strings and a more
# robust color signature (type, rgb, indexed, theme, tint) to match theme/tinted
# colors that may not share the exact RGB.
ADDITIONAL_EMPTY_ARGB: set[str] = set()
ADDITIONAL_EMPTY_SIGS: set[Tuple[Any, Any, Any, Any, Any]] = set()

# Dynamic additional wall colors (treated as walls across the sheet)
ADDITIONAL_WALL_ARGB: set[str] = set()
ADDITIONAL_WALL_SIGS: set[Tuple[Any, Any, Any, Any, Any]] = set()
ADDITIONAL_WALL_RGBS: List[Tuple[int, int, int]] = []
ADDITIONAL_WALL_THEME_IDS: set[int] = set()
ADDITIONAL_WALL_INDEXED_IDS: set[int] = set()


def _color_signature(cell: Cell) -> Optional[Tuple[Any, Any, Any, Any, Any]]:
    fill = cell.fill
    if fill is None or fill.patternType is None:
        return None
    color = fill.start_color
    if color is None:
        return None
    return (getattr(color, "type", None), getattr(color, "rgb", None), getattr(color, "indexed", None), getattr(color, "theme", None), getattr(color, "tint", None))


def _rgb_triplet(argb: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if not argb or len(argb) not in (6, 8):
        return None
    # Strip alpha if present (ARGB → RGB)
    hexrgb = argb[-6:]
    try:
        r = int(hexrgb[0:2], 16)
        g = int(hexrgb[2:4], 16)
        b = int(hexrgb[4:6], 16)
        return (r, g, b)
    except Exception:
        return None


def _rgb_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> int:
    return abs(c1[0]-c2[0]) + abs(c1[1]-c2[1]) + abs(c1[2]-c2[2])


def _color_to_argb(cell: Cell) -> Optional[str]:
    """Return the ARGB hex string for the cell fill color if present.

    openpyxl may encode colors as RGB, indexed palette, or theme-based.
    This helper normalizes to an ARGB string when possible.
    """
    fill = cell.fill
    if fill is None or fill.patternType is None:
        return None

    color = fill.start_color
    if color is None:
        return None

    # RGB specified directly
    if getattr(color, "type", None) == "rgb" and color.rgb:
        return color.rgb.upper()

    # Indexed palette
    if getattr(color, "type", None) == "indexed" and color.indexed is not None:
        try:
            argb = COLOR_INDEX[color.indexed]
            return (argb or "").upper() or None
        except Exception:
            return None

    # Theme-based colors are hard to resolve without workbook theme; ignore
    return None


def _is_wall_fill(cell: Cell) -> bool:
    """Heuristic: consider filled grey/brown cells as walls.

    Returns True if the cell's fill color matches one of the configured
    grey/brown ARGB values.
    """
    argb = _color_to_argb(cell)
    if not argb:
        return False
    # Dynamic WALL has highest priority
    sig = _color_signature(cell)
    if (argb in ADDITIONAL_WALL_ARGB) or (sig and sig in ADDITIONAL_WALL_SIGS):
        return True
    # Dynamic EMPTY has next priority (prevent palette-based wall)
    if (argb in ADDITIONAL_EMPTY_ARGB) or (sig and sig in ADDITIONAL_EMPTY_SIGS):
        return False
    # Static palettes
    if (argb in GREY_HEXES) or (argb in BROWN_HEXES):
        return True
    # Fuzzy RGB distance to capture slight theme/tint differences
    rgb = _rgb_triplet(argb)
    if rgb and any(_rgb_distance(rgb, ref) <= 48 for ref in ADDITIONAL_WALL_RGBS):
        return True
    # Theme/indexed id match
    color = cell.fill.start_color if cell.fill else None
    if color is not None:
        if getattr(color, "type", None) == "theme" and getattr(color, "theme", None) in ADDITIONAL_WALL_THEME_IDS:
            return True
        if getattr(color, "type", None) == "indexed" and getattr(color, "indexed", None) in ADDITIONAL_WALL_INDEXED_IDS:
            return True
    return False


def _normalize_token(val: Optional[str]) -> str:
    """Normalize cell textual content for symbol matching.

    - Strips whitespace
    - Upper-cases ASCII letters
    - Returns empty string for None
    """
    if val is None:
        return ""
    return str(val).strip().upper()


def symbol_to_minigrid(token: str) -> int:
    """Map a normalized token to a MiniGrid integer id.

    The mapping follows project conventions while reusing existing ids
    to remain compatible with `gym_minigrid.numpymap` rendering:
    - A, B, C: different victim goals (81/82/83)
    - X: collapse plate / threat collapse → hazard (lava id 9)
    - D: falling rubble → hazard (lava id 9)
    - T: freezing threat → hazard (lava id 9)
    - P: victim detection plate → interactable plate (box id 255)
    - F: object → generic object (box id 255)
    - R, RR, RRR: rubble layers → walls (4 or 30 for heavier rubble)
    - empty/other: walkable empty (1)
    """
    if token == "A":
        return GOAL_A
    if token == "B":
        return GOAL_B
    if token == "C":
        return GOAL_C
    if token in {"X"}:  # red collapse (render as red box rather than lava)
        return BOX_RED
    if token == "D":  # falling rubble → dark blue box
        return BOX_DARK_BLUE
    if token == "T":  # freezing threat → treat as hazard
        return LAVA
    if token == "P":  # victim detection plate → light blue
        return BOX_LIGHT_BLUE
    if token == "F":  # object → light brown
        return BOX
    if token in {"RRR", "RR", "R"}:  # rubble → light grey wall
        return WALL_LIGHT
    return EMPTY


def find_content_origin(sheet) -> Tuple[int, int]:
    """Best-effort detection of the first map row/column.

    Many mapping sheets include header rows/columns. We scan top-left area
    to find the earliest row and column that contain either a known symbol
    or a filled wall cell.

    Returns (row_start, col_start) as 1-based indices for openpyxl.
    """
    max_scan_rows = min(50, sheet.max_row)
    max_scan_cols = min(50, sheet.max_column)

    row_start = None
    col_start = None

    for r in range(1, max_scan_rows + 1):
        for c in range(1, max_scan_cols + 1):
            cell = sheet.cell(row=r, column=c)
            token = _normalize_token(cell.value)
            if token in {"A", "B", "C", "X", "D", "P", "R", "RR", "RRR", "T", "F"} or _is_wall_fill(cell):
                row_start = r if row_start is None else min(row_start, r)
                col_start = c if col_start is None else min(col_start, c)
    if row_start is None or col_start is None:
        # Fallback to 1,1 if detection fails
        row_start = 1
        col_start = 1
    # Override by fixed range origin C5 if present
    return EXCEL_RANGE_START[0], EXCEL_RANGE_START[1]


def parse_saturn_sheet(sheet_name: str, save_basename: str) -> None:
    """Parse one Saturn sheet into a MiniGrid numpy array and save it.

    Parameters
    ----------
    sheet_name: str
        Name of the sheet in the workbook to parse.
    save_basename: str
        Basename for the output files (without extension). Two files are
        produced: `<basename>.npy` and `raw_map_state_<suffix>.npy`.
    """
    wb = load_workbook(WORKBOOK_PATH, data_only=True)
    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Sheet '{sheet_name}' not found in {WORKBOOK_PATH}")
    sheet = wb[sheet_name]

    # Populate dynamic empty-color exceptions from designated cells
    # As requested: use colors from AT11, CB11, DJ11 as canonical EMPTY colors
    from_cells = ("AT", "CB", "DJ")
    ADDITIONAL_EMPTY_ARGB.clear()
    ADDITIONAL_EMPTY_SIGS.clear()
    ADDITIONAL_WALL_ARGB.clear()
    ADDITIONAL_WALL_SIGS.clear()
    ADDITIONAL_WALL_THEME_IDS.clear()
    ADDITIONAL_WALL_INDEXED_IDS.clear()
    # IMPORTANT: reset RGB reference list per sheet to avoid leakage across runs
    ADDITIONAL_WALL_RGBS.clear()
    for col in from_cells:
        try:
            ci = column_index_from_string(col)
            # sample a few nearby rows to capture theme/tint variants
            for rr in (11, 12, 10):
                cell = sheet.cell(row=rr, column=ci)
                argb = _color_to_argb(cell)
                if argb:
                    ADDITIONAL_EMPTY_ARGB.add(argb)
                sig = _color_signature(cell)
                if sig:
                    ADDITIONAL_EMPTY_SIGS.add(sig)
        except Exception:
            pass

    # Additionally: sample AG10 as a canonical WALL color and include it
    try:
        ag_col = column_index_from_string('AG')
        for rr in (9, 10, 11, 12):
            cell = sheet.cell(row=rr, column=ag_col)
            argb = _color_to_argb(cell)
            if argb:
                ADDITIONAL_WALL_ARGB.add(argb)
                rgb = _rgb_triplet(argb)
                if rgb:
                    ADDITIONAL_WALL_RGBS.append(rgb)
            sig = _color_signature(cell)
            if sig:
                ADDITIONAL_WALL_SIGS.add(sig)
            color = cell.fill.start_color if cell.fill else None
            if color is not None:
                if getattr(color, "type", None) == "theme" and getattr(color, "theme", None) is not None:
                    ADDITIONAL_WALL_THEME_IDS.add(getattr(color, "theme", None))
                if getattr(color, "type", None) == "indexed" and getattr(color, "indexed", None) is not None:
                    ADDITIONAL_WALL_INDEXED_IDS.add(getattr(color, "indexed", None))
    except Exception:
        pass

    # Initialize grid (row-major: y, x)
    grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int32) + EMPTY

    row_start, col_start = find_content_origin(sheet)

    # Compute Excel end column for EK (use openpyxl translator)
    from openpyxl.utils import column_index_from_string
    excel_end_col = column_index_from_string('EK')
    excel_rows = EXCEL_RANGE_END[0] - EXCEL_RANGE_START[0] + 1
    excel_cols = excel_end_col - EXCEL_RANGE_START[1] + 1

    # Validate computed size matches grid size; if not, crop to min
    h = min(GRID_HEIGHT, excel_rows)
    w = min(GRID_WIDTH, excel_cols)

    for r in range(h):
        for c in range(w):
            cell = sheet.cell(row=row_start + r, column=col_start + c)
            token = _normalize_token(cell.value)

            # First, handle dynamic WALL/EMPTY overrides by color
            argb = _color_to_argb(cell)
            sig = _color_signature(cell)
            # Treat any cell matching AG10's color (ARGB/signature/theme/indexed/fuzzy RGB)
            # as WALL prior to any EMPTY override
            if (argb in ADDITIONAL_WALL_ARGB) or (sig and sig in ADDITIONAL_WALL_SIGS):
                grid[r, c] = WALL
                continue
            # Fuzzy RGB/theme/indexed checks for AG10 wall color match
            if argb:
                rgb = _rgb_triplet(argb)
                if (rgb and any(_rgb_distance(rgb, ref) <= 48 for ref in ADDITIONAL_WALL_RGBS)):
                    grid[r, c] = WALL
                    continue
            color = cell.fill.start_color if cell.fill else None
            if color is not None:
                if getattr(color, "type", None) == "theme" and getattr(color, "theme", None) in ADDITIONAL_WALL_THEME_IDS:
                    grid[r, c] = WALL
                    continue
                if getattr(color, "type", None) == "indexed" and getattr(color, "indexed", None) in ADDITIONAL_WALL_INDEXED_IDS:
                    grid[r, c] = WALL
                    continue
            if (argb in ADDITIONAL_EMPTY_ARGB) or (sig and sig in ADDITIONAL_EMPTY_SIGS):
                grid[r, c] = EMPTY
                continue

            # Otherwise: prefer explicit symbol; if still empty, infer by fill
            val = symbol_to_minigrid(token)
            if val == EMPTY and _is_wall_fill(cell):
                val = WALL
            grid[r, c] = val

    # Explicitly enforce the provided world coordinate (-2195, -6) from AG10
    # Map world (x,y) to array indices: col = x - TOP_LEFT[0], row = y - TOP_LEFT[1]
    try:
        enforce_x, enforce_y = -2195, -6
        rr = enforce_y - TOP_LEFT[1]
        cc = enforce_x - TOP_LEFT[0]
        if 0 <= rr < GRID_HEIGHT and 0 <= cc < GRID_WIDTH:
            grid[rr, cc] = WALL
    except Exception:
        pass

    # Add a solid wall border for safety (consistent with legacy behavior)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    # Build raw_map_state: walkable=1, walls=4; goals/boxes remain walkable
    raw_map_state = (grid.copy()).astype(np.int32)
    raw_map_state[raw_map_state == BOX] = EMPTY
    raw_map_state[raw_map_state == GOAL_A] = EMPTY
    raw_map_state[raw_map_state == GOAL_B] = EMPTY
    raw_map_state[raw_map_state == GOAL_C] = EMPTY
    raw_map_state[raw_map_state == LAVA] = WALL  # keep generic hazards blocked if any
    raw_map_state[raw_map_state == WALL_HEAVY] = WALL
    raw_map_state[raw_map_state == WALL_LIGHT] = WALL

    # Save outputs
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    out_map = RESOURCES_DIR / f"{save_basename}.npy"
    np.save(out_map, grid)

    out_state = RESOURCES_DIR / f"raw_map_state_{save_basename.split('_')[0].lower()}.npy"
    np.save(out_state, raw_map_state)

    print(f"Saved: {out_map}")
    print(f"Saved: {out_state}")


def main() -> None:
    """Entry point: parse both Saturn maps and save outputs.

    This function is idempotent and will overwrite existing files.
    """
    if not WORKBOOK_PATH.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK_PATH}")

    sheet_to_out: Dict[str, str] = {
        "SaturnA_2.3": "SaturnA_2_3",
        "SaturnB_2.3": "SaturnB_2_3",
    }

    for sheet_name in SHEET_NAMES:
        parse_saturn_sheet(sheet_name, sheet_to_out[sheet_name])


if __name__ == "__main__":
    main()
