# Action Extraction Pipeline (ASIST New Study)

This document describes how to prepare inputs, run the extractor at 0.1 s sampling, and understand the outputs. It also summarizes the action codes, topics used, and region mapping logic.

## 1. Prerequisites
- Python 3.9+ with the project environment installed.
- Semantic map file:
  - `data/map_excel/Saturn_2.6_3D_sm_v1.0.json` (rectangle bounds with region IDs)
- Trial metadata files:
  - Place raw metadata `.metadata` files under `data/trials/` (see “2. Prepare trials”).

## 2. Prepare trials
1) Create the directory if it does not exist:
   - `data/trials/`
2) Download the raw trial message files (e.g., `HSRData_TrialMessages_Trial-T0006xx_*.metadata`) into `data/trials/`. These files are newline-delimited JSON records.
3) The extractor will automatically scan this directory by default glob:  
   `data/trials/HSRData_TrialMessages_Trial-*.metadata`

## 3. Run the extractor
Use the default 0.1 s sampling, starting after the mission timer reaches 15:00.

```bash
python utils/new_study_action_extractor.py \
  --regions data/map_excel/Saturn_2.6_3D_sm_v1.0.json \
  --output  data/new_study_actions
```

Notes:
- By default, the extractor processes all trials matched by `--metadata` (default glob above).
- Allowed missions: `Saturn_A`, `Saturn_B`, `Saturn_C`, `Saturn_D`. Trials with other missions are skipped.
- Topics considered:
  - `observations/state`
  - `observations/events/player/victim_picked_up`
  - `observations/events/player/victim_placed`
  - `observations/events/player/triage`
  - `observations/events/player/rubble_destroyed`
  - `observations/events/player/location` (used only to improve resolution where available)
  - `trial` (used for role/participant mapping)
- The event `observations/events/server/victim_evacuated` is intentionally ignored.
- Entries with `mission_timer == "Mission Timer not initialized."` are ignored.

## 4. Action space and codes
- Movement (8 directions, codes 0–7):
  - 0: move left (−1, 0)
  - 1: move right (+1, 0)
  - 2: move up (0, −1)
  - 3: move down (0, +1)
  - 4: move up-left (−1, −1)
  - 5: move up-right (+1, −1)
  - 6: move down-left (−1, +1)
  - 7: move down-right (+1, +1)
- Other actions:
  - 8: no action (idle)
  - 9: triage
  - 10: clear rubble
  - 11: pickup victim
  - 12: drop victim

## 5. Region mapping
- The extractor maps `(x, z)` to a region ID using axis-aligned rectangle bounds from `Saturn_2.6_3D_sm_v1.0.json`.
- When an exact match is not found, the extractor probes alternate points (floored-tile center) to reduce “unknown” classifications.
- If still unknown, the extractor falls back to the player’s last known region for continuity.

## 6. Output directory structure
For each trial, the extractor creates a subdirectory named:  
`<TRIAL_TAG>_<MISSION>` e.g., `T000602_Saturn_B`

Files:
- `summary.json`:
  - `trial_id`, `mission`, `mission_source`
  - `roles`: per-role player list with participant IDs
  - `victim_count`
- `actions_engineer.json`, `actions_medic.json`, `actions_transporter.json`:
  - `trial_id`, `mission`, `role`
  - `players`, `participant_ids`, `player_participants`
  - `actions`: integer action codes (see §4)
  - `action_labels`: human-readable action labels
  - `mission_timers`: “MM : SS” per step (post-15:00)
  - `regions`: region ID per step
  - `x_coordinates`, `z_coordinates`: position components per step (may be None when unavailable)
- `victims.json`:
  - `timeline`: aligned snapshots (0.1 s cadence) with:
    - `timestamp`
    - `victims`: array of victim states with fields:
      - `victim_id`
      - `position`: `{x, y, z}` (falls back to last known when missing)
      - `region`: region ID (falls back to last known when unknown)
      - `triaged`: boolean

## 7. Time alignment & sampling
- The extractor uses a unified 0.1 s clock per role and aligns discrete events by overwriting the nearest future state frame.
- Large coordinate jumps detected between adjacent states are treated as idles in that step (to avoid encoding multi-step teleports as a single move). Downstream consumers may apply path interpolation if desired.

## 8. Troubleshooting
- If outputs contain many `unknown` regions:
  - Verify coordinates fall inside the map bounds.
  - Ensure `Saturn_2.6_3D_sm_v1.0.json` is present and correctly formatted.
- If `actions_*.json` arrays have mismatched lengths:
  - Re-run extraction; all arrays must be aligned one-to-one.


