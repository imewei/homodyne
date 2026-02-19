"""One-off helper to probe laminar CMC log-density at chosen inits.

Usage (from repo root):
    python scripts/laminar_logprob_probe.py \
        --config /home/wei/Documents/Projects/data/C020/homodyne_laminar_flow_config.yaml \
        --cache  /home/wei/Documents/Projects/data/C020/cached_c2_flow_q0.0054_frames_1000_2000.npz

It builds a single shard (first 20k points of the first selected phi) using the
same angle filtering as the config, then evaluates the NumPyro model log-density
twice: (a) current config initial_values, (b) a tamed variant with reduced D0
and alpha. It reports whether the logprob is finite/NaN/-inf to explain sampling
collapse.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_density

from homodyne.config.manager import ConfigManager
from homodyne.config.parameter_space import ParameterSpace
from homodyne.optimization.cmc.model import xpcs_model_scaled
from homodyne.optimization.cmc.scaling import compute_scaling_factors


def load_shard(cache_path: Path, angle_ranges: list) -> dict:
    """Load cached C2 and build a single-shard slice matching CMC prep.

    Returns dict with data/t1/t2/phi_unique/phi_indices and q, dt, noise_scale.
    """
    f = np.load(cache_path)

    phi_all = f["phi_angles_list"]
    # angle filtering
    mask_phi = np.zeros_like(phi_all, dtype=bool)
    for lo, hi in angle_ranges:
        mask_phi |= (phi_all >= lo) & (phi_all <= hi)
    sel_idx = np.where(mask_phi)[0]
    phi_sel = phi_all[sel_idx]

    # drop t=0
    c2 = f["c2_exp"][sel_idx][:, 1:, 1:]
    t1_raw = f["t1"][1:]
    t2_raw = f["t2"][1:]
    T1, T2 = np.meshgrid(t1_raw, t2_raw)

    phi_repeat = np.repeat(phi_sel[:, None, None], c2.shape[1], axis=1)
    phi_repeat = np.repeat(phi_repeat, c2.shape[2], axis=2)

    pooled_data = c2.reshape(-1)
    phi_flat = phi_repeat.reshape(-1)
    t1_flat = T1.reshape(-1)
    t2_flat = T2.reshape(-1)

    phi_unique = np.unique(phi_flat)

    # pick first phi shard, first 20k points
    mask_angle0 = phi_flat == phi_unique[0]
    idx = np.where(mask_angle0)[0][:20000]

    data = pooled_data[idx]
    t1 = t1_flat[idx]
    t2 = t2_flat[idx]
    phi_indices = np.zeros_like(idx)  # all belong to the first phi in phi_unique

    noise_scale = 0.0662  # from run log estimation

    return {
        "data": jnp.array(data),
        "t1": jnp.array(t1),
        "t2": jnp.array(t2),
        "phi_unique": jnp.array(phi_unique),
        "phi_indices": jnp.array(phi_indices),
        "q": float(f["wavevector_q_list"][0]),
        "L": 2e6,
        "dt": 0.1,
        "analysis_mode": "laminar_flow",
        "n_phi": len(phi_unique),
        "time_grid": None,
        "noise_scale": noise_scale,
    }


def build_init_dict(
    base_init: dict[str, float],
    contrast: float,
    offset: float,
    param_space: ParameterSpace,
    scalings: dict,
) -> dict[str, float]:
    """Fill missing init values with midpoint of bounds, converted to z-space.

    xpcs_model_scaled samples in z-space: z ~ N(0, 1), then transforms
    P = center + scale * z. We invert: z = (P - center) / scale.
    """

    def mid(name: str) -> float:
        lo, hi = param_space.get_bounds(name)
        return (lo + hi) / 2.0

    def to_z(name: str, value: float) -> float:
        s = scalings[name]
        return (value - s.center) / s.scale

    physical = {
        "contrast_0": contrast,
        "contrast_1": contrast,
        "contrast_2": contrast,
        "offset_0": offset,
        "offset_1": offset,
        "offset_2": offset,
        "D0": float(base_init.get("D0", mid("D0"))),
        "alpha": float(base_init.get("alpha", mid("alpha"))),
        "D_offset": float(base_init.get("D_offset", mid("D_offset"))),
        "gamma_dot_t0": float(base_init.get("gamma_dot_t0", mid("gamma_dot_t0"))),
        "beta": float(base_init.get("beta", mid("beta"))),
        "gamma_dot_t_offset": float(
            base_init.get("gamma_dot_t_offset", mid("gamma_dot_t_offset"))
        ),
        "phi0": float(base_init.get("phi0", mid("phi0"))),
    }
    # Convert to z-space for xpcs_model_scaled
    init = {f"{name}_z": to_z(name, val) for name, val in physical.items()}
    init["sigma"] = 0.0662
    return init


def main(args: argparse.Namespace) -> None:
    cm = ConfigManager(str(args.config))
    config = cm.config or {}
    ps = ParameterSpace.from_config(config, analysis_mode="laminar_flow")

    # angle filter ranges from config; fallback to defaults used in run
    angle_filter = config.get("optimization", {}).get("angle_filtering", {})
    ranges = angle_filter.get("ranges", [[-10.0, 10.0], [85.0, 95.0]])
    angle_ranges = [(float(lo), float(hi)) for lo, hi in ranges]

    shard = load_shard(Path(args.cache), angle_ranges)

    scalings = compute_scaling_factors(ps, shard["n_phi"], "laminar_flow")

    init_cfg = config.get("optimization", {}).get("mcmc", {}).get("initial_values", {})
    contrast = float(init_cfg.get("contrast", 0.5))
    offset = float(init_cfg.get("offset", 1.0))

    init_a = build_init_dict(init_cfg, contrast, offset, ps, scalings)
    init_b = init_a.copy()
    # Override in z-space: z = (physical - center) / scale
    s_d0 = scalings["D0"]
    s_alpha = scalings["alpha"]
    init_b["D0_z"] = (args.tamed_d0 - s_d0.center) / s_d0.scale
    init_b["alpha_z"] = (args.tamed_alpha - s_alpha.center) / s_alpha.scale

    for label, init in [("config", init_a), ("tamed", init_b)]:
        try:
            logp, _ = log_density(
                xpcs_model_scaled,
                model_args=(),
                model_kwargs={**shard, "parameter_space": ps},
                params=init,
            )
            finite = jnp.isfinite(logp)
            print(f"{label} init: logp={float(logp):.3f}, finite={bool(finite)}")
        except Exception as exc:  # noqa: BLE001
            print(f"{label} init: log_density failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Probe laminar log-density at given inits"
    )
    parser.add_argument("--config", required=True, type=Path, help="YAML config path")
    parser.add_argument("--cache", required=True, type=Path, help="cached C2 npz path")
    parser.add_argument("--tamed-d0", type=float, default=2000.0)
    parser.add_argument("--tamed-alpha", type=float, default=-0.5)
    args = parser.parse_args()
    main(args)
