"""
Visualize generated MiniGrid maps by rendering through the actual
gym-minigrid environment, not via raw numpy imshow.

The script constructs a `MiniGrid-NumpyMap-v0` environment using each
generated map and renders the full-grid view to PNGs. This ensures the
visual output matches how the engine interprets IDs, colors, and walls.

Run:
    python -m utils.visualize_maps
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from gym_minigrid.envs.numpymap import NumpyMap


RESOURCES_DIR = (Path(__file__).parent.parent / "gym_minigrid" / "envs" / "resources").resolve()
OUT_DIR = RESOURCES_DIR / "vis"


def render_env_from_map(base_name: str, tile_size: int = 8) -> None:
    """Create a NumpyMap env for the given map and save a rendered PNG.

    Parameters
    ----------
    base_name: str
        Basename of the `.npy` map within RESOURCES_DIR.
    tile_size: int
        Pixel size per tile for rendering.
    """
    npy_path = RESOURCES_DIR / f"{base_name}.npy"
    numpy_array = np.load(npy_path)

    # Instantiate the environment class directly to avoid importing
    # gym wrappers that may not be compatible with local gym versions.
    # Place the agent at a valid inside-cell; (1,1) lies just inside the outer wall
    env = NumpyMap(numpy_array=numpy_array, agent_pos=(1, 1), agent_dir=0, max_steps=1_000)
    env.seed(0)
    env.reset()

    img = env.render(mode='rgb_array', tile_size=tile_size)

    import matplotlib.pyplot as plt  # imported lazily to keep deps light for callers
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{base_name}.png"
    plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    plt.axis('off')
    plt.imshow(img)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    env.close()
    print(f"Saved visualization: {out_path}")


def main() -> None:
    # Render both high-level maps and raw state overlays for reference
    for base in [
        "SaturnA_2_3",
        "SaturnB_2_3",
        "SaturnC_2_3",
        "SaturnD_2_3",
        "raw_map_state_saturna",
        "raw_map_state_saturnb",
        "raw_map_state_saturnc",
        "raw_map_state_saturnd",
    ]:
        render_env_from_map(base)


if __name__ == "__main__":
    main()


