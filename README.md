# NGT Plan Recognition – Saturn/ASIST Extension

## Project Lineage

This repository continues the work that began in the [`ualiangzhang/tomcat-il`](https://github.com/ualiangzhang/tomcat-il) codebase and re-implements the modeling approach described in the AAAI-FSS 2021 paper *“Using Features at Multiple Temporal and Spatial Resolutions to Predict Human Behavior in Real Time.”*[^paper]

[^paper]: Liang Zhang, Justin Lieffers, Adarsh Pyarelal. Computational Theory of Mind for Human-Machine Teams, Lecture Notes in Computer Science, vol. 13775, 2023\. https://link.springer.com/chapter/10.1007/978-3-031-21671-8_13

## Program Context

We are in the process of generalizing the original ASIST Study 1 pipeline to cover Studies 2 and 3:

- **Team Size Expansion** – from single-operator missions to full three-person rescue teams with heterogeneous roles.
- **Multimodal Observations** – synchronized language transcriptions, mission metadata, and event logs for richer teammate modeling.
- **Predictive Objectives** – estimating individual and team strategies, proactive intent recognition, and mission-phase forecasting inside MiniGrid-based simulators.

## Current Capabilities

### High-Fidelity Map Processing

- `gym_minigrid/map_parser_for_excel.py` ingests Saturn facility layouts from Excel blueprints, keeping the engineering wall layers intact while overlaying interactive elements (victims, signals, hazards) from curated CSV blocks (`data/map_excel/MapBlocks_Saturn*.csv`).
- The parser produces MiniGrid-compatible `.npy` artifacts plus “raw walkability” tensors for Saturn A–D. Consistency checks ensure each Victim A/B/C zone is tagged with the correct goal tile.
- `utils/visualize_maps.py` batches renderings of both the gameplay grids and raw walkability states to `gym_minigrid/envs/resources/vis/*.png` for rapid QA.

### Action Understanding & Behavior Extraction

- `utils/new_study_action_extractor.py` normalizes metadata dumps across mission types, now including `Saturn_C` and `Saturn_D`, and captures initial victim context on first sighting for downstream ToM features.
- `utils/replay_minigrid_actions.py` offers a scaffold for deterministic replays against the generated maps, enabling side-by-side comparisons between logged trajectories and model rollouts.

### Connectivity Reasoning

- `utils/build_saturn_connectivity.py` fuses geometric bounds (`data/map_excel/Saturn_2.6_3D_sm_v1.0.json`) with mission adjacency (`data/Saturn_adjacency_by_level.json`) to infer north/south/east/west relations per level. The resulting directional graph is stored in `data/saturn_directional_connectivity.json`.
- The same script renders per-level diagnostics (`gym_minigrid/envs/resources/vis/saturn_connectivity_level_{1,2}.png`) so that spatial components, arrows, and direction legends can be inspected visually.

### Dataset & Utilities Snapshot

- `data/new_study_actions*/` – normalized action and event logs partitioned for experimentation.
- `gym_minigrid/envs/resources/` – canonical MiniGrid assets, model-ready `.npy` maps, and visualization products.
- `requirements.txt` – Python dependencies (MiniGrid, NumPy, Matplotlib, OpenPyXL, etc.) for reproducible builds.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Rebuild Saturn maps and visuals
python gym_minigrid/map_parser_for_excel.py
python utils/visualize_maps.py

# Recompute directional connectivity (JSON + PNGs)
python utils/build_saturn_connectivity.py \
  --output data/saturn_directional_connectivity.json \
  --vis-dir gym_minigrid/envs/resources/vis
```

## Roadmap Highlights

- Integrate three-operator coordination policies and conversational cues directly into the MiniGrid rollouts.
- Extend action replay tooling into evaluators for policy distillation and imitation learning.
- Automate validation suites that cross-check Excel, CSV, and adjacency sources to catch regressions as new Saturn variants arrive.

For questions or contributions, please open an issue or contact the maintainers.

