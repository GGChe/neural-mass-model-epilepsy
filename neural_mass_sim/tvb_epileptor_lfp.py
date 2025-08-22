#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate epilepsy LFP-like signals (interictal & ictal) with TVB Epileptor.

Outputs (created under ./outputs):
  - epileptor_interictal.csv/.png
  - epileptor_ictal.csv/.png

CSV columns: t_s, lfp_x2_minus_x1, z
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tvb.simulator.lab import models, integrators, coupling, monitors, simulator, connectivity
from tvb.simulator import noise


# ---------------- Connectivity ----------------

def build_single_node_connectivity():
    """Create a minimal 1-node Connectivity (no tvb-data dependency)."""
    conn = connectivity.Connectivity()
    conn.number_of_regions = 1
    conn.weights        = np.array([[0.0]], dtype=float)
    conn.tract_lengths  = np.array([[0.0]], dtype=float)
    conn.region_labels  = np.array(["Region1"], dtype="U128")
    conn.centres        = np.array([[0.0, 0.0, 0.0]], dtype=float)
    conn.areas          = np.array([1.0], dtype=float)
    conn.cortical       = np.array([True], dtype=bool)
    conn.configure()
    return conn


# ---------------- Helpers for TVB monitor output ----------------

def _normalize_run_output(run_out):
    """
    TVB sim.run(...) may return:
      - (time, data) for a single monitor, or
      - a list/tuple of (time, data) per monitor (with possible None entries).
    Return (t_ms (1D), y (ndarray)).
    """
    if isinstance(run_out, tuple) and len(run_out) == 2:
        t_ms, y = run_out
    else:
        t_ms, y = None, None
        for item in run_out:
            if item is None:
                continue
            if isinstance(item, tuple) and len(item) == 2:
                t_ms, y = item
                break
    if t_ms is None or y is None:
        raise RuntimeError("Simulator returned no data. Check monitors/period/duration.")
    return np.asarray(t_ms).squeeze(), np.asarray(y)


def _extract_Tx2(y, n_vars=2):
    """
    Convert Raw monitor output 'y' to shape (T, 2) where columns are (x2-x1, z).
    Handles shapes like (T,2,1,1), (T,1,1,2), (T,1,2), (T,2), etc.
    Assumes axis 0 is time; finds the axis whose length == n_vars and brings it to axis 1.
    Selects the first slice over remaining axes.
    """
    y = np.asarray(y)
    if y.ndim == 2:
        return y if y.shape[1] == n_vars else y.T

    var_axis = None
    for ax in range(1, y.ndim):
        if y.shape[ax] == n_vars:
            var_axis = ax
            break
    if var_axis is None:
        if y.shape[0] == n_vars:
            y = np.swapaxes(y, 0, 1)  # make vars axis 1
            var_axis = 1
        else:
            raise RuntimeError(f"Cannot find variables axis of length {n_vars}; y.shape={y.shape}")

    y2 = np.moveaxis(y, var_axis, 1)   # -> (T, 2, ...)
    T = y2.shape[0]
    y2 = y2.reshape(T, n_vars, -1)     # flatten remaining axes
    return y2[:, :, 0]                 # first region/realization -> (T, 2)


# ---------------- Core simulation ----------------

def simulate_epileptor(mode="ictal",
                       duration_s=60.0,
                       fs=1000.0,
                       warmup_s=5.0,
                       seed=7):
    """
    Generate LFP-like signals from TVB Epileptor.

    mode:
      - "ictal": sustained seizure-like oscillations
      - "interictal": sporadic spikes driven by noise on population 2

    Returns (t_s, lfp, z) AFTER discarding 'warmup_s' seconds.
    """
    rng = np.random.default_rng(seed)

    # --- Model: variables of interest ---
    mdl = models.Epileptor()
    mdl.variables_of_interest = ("x2 - x1", "z")

    # --- Parameter presets that are reliably non-flat ---
    if mode == "ictal":
        # Push into seizure-prone regime (sustained oscillations)
        mdl.x0 = np.array([-1.60])      # epileptogenic
        mdl.r  = np.array([0.00015])    # slower z -> longer ictal plateaus
        # modest additive noise on pop-2 states
        noise_D = np.array([0, 0, 0, 2e-4, 2e-4, 0], dtype=float)
    else:  # interictal
        # Near the threshold but in the interictal basin; noise triggers spikes
        mdl.x0 = np.array([-1.85])
        mdl.r  = np.array([0.00030])
        # slightly stronger noise on pop-2 states to elicit spikes
        noise_D = np.array([0, 0, 0, 4e-4, 4e-4, 0], dtype=float)

    # --- Integrator & sampling ---
    dt = 0.05e-3                                    # 0.05 ms integration step
    integ = integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=noise_D))
    period = max(1, int(round((1.0 / fs) / dt)))    # Raw monitor sampling period
    mon = monitors.Raw(period=period)

    # --- Connectivity & coupling ---
    conn = build_single_node_connectivity()
    coup = coupling.Difference(a=np.array([1.0], dtype=float))

    # --- Simulator ---
    sim = simulator.Simulator(
        model=mdl, connectivity=conn, coupling=coup, integrator=integ, monitors=(mon,)
    ).configure()

    # --- Run & normalize outputs ---
    run_out = sim.run(simulation_length=duration_s)
    t_ms, y = _normalize_run_output(run_out)
    Y = _extract_Tx2(y, n_vars=2)

    t_s = (t_ms * 1e-3).astype(float)
    lfp = Y[:, 0]
    z   = Y[:, 1]

    # --- Discard initial transient/warmup ---
    if warmup_s > 0.0:
        keep = t_s >= float(warmup_s)
        t_s, lfp, z = t_s[keep], lfp[keep], z[keep]

    return t_s, lfp, z


# ---------------- CLI: run both modes, save CSV & PNG ----------------

if __name__ == "__main__":
    outdir = pathlib.Path("outputs")
    outdir.mkdir(parents=True, exist_ok=True)

    for mode in ("interictal", "ictal"):
        # Longer duration so events clearly appear
        t, lfp, z = simulate_epileptor(
            mode=mode,
            duration_s=60.0,   # realistic horizon
            fs=1000.0,
            warmup_s=5.0,      # drop initial transient for cleaner plots/CSV
            seed=7
        )

        # Save CSV
        csv_path = outdir / f"epileptor_{mode}.csv"
        pd.DataFrame(
            {"t_s": t, "lfp_x2_minus_x1": lfp, "z": z}
        ).to_csv(csv_path, index=False)

        # Save PNG
        plt.figure(figsize=(12, 4))
        plt.plot(t, lfp, lw=0.9, label="LFP (x2 - x1)")
        z_sc = (z - z.mean()) / (z.std() + 1e-9)
        plt.plot(t, z_sc, lw=0.7, alpha=0.7, label="z (z-scored)")
        plt.title(f"Epileptor — {mode}")
        plt.xlabel("Time [s]"); plt.ylabel("Amplitude (a.u.)")
        plt.legend()
        plt.tight_layout()
        fig_path = outdir / f"epileptor_{mode}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

        print(f"[{mode}] Saved: {csv_path}")
        print(f"[{mode}] Saved: {fig_path}")
