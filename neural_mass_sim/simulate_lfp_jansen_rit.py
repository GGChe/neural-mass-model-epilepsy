#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jansen-Rit (PyRates) synthetic LFP with interictal spikes and a SINGLE clear ictal epoch.

NO FILTERING. The script:
- keeps the raw signal,
- adds a baseline-aligned copy for the ictal figure and CSV (so ictal and baseline share the same mean),
- uses integrate()-based probing to resolve output/input variable paths robustly.

Outputs (./outputs):
  jr_lfp_interictal.csv/.png
  jr_lfp_ictal.csv/.png
Each CSV contains: t_s, lfp_raw, lfp_aligned, ictal_mask (0/1).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from pyrates import integrate

def ou_noise(T, dt, mu=120.0, tau=0.05, sigma=25.0, seed=7):
    """Ornstein-Uhlenbeck process (pulses/s), non-negative."""
    rng = np.random.default_rng(seed)
    n = int(np.round(T / dt))
    x = np.empty(n); x[0] = mu
    a = dt / tau
    s = np.sqrt(2.0 * (sigma**2) * dt / tau)
    for k in range(n - 1):
        x[k + 1] = x[k] + a * (mu - x[k]) + s * rng.standard_normal()
    return np.clip(x, 0.0, None)

def alpha_pulse_kernel(dt, tau=0.010, length=0.100):
    """Alpha kernel h(t) = (t/tau) * exp(1 - t/tau), unit peak."""
    t = np.arange(0.0, length + dt/2, dt)
    h = (t / tau) * np.exp(1.0 - t / tau)
    m = np.max(h)
    return h / m if m > 0 else h

def make_interictal_p(T, dt, *, mu=150.0, sigma=30.0,
                      lam_per_min=10.0, pulse_amp=800.0, tau=0.010, seed=7):
    """OU baseline plus sparse alpha-shaped pulses to elicit interictal spikes."""
    p = ou_noise(T, dt, mu=mu, tau=0.05, sigma=sigma, seed=seed)
    n = p.size
    h = alpha_pulse_kernel(dt, tau=tau, length=6 * tau)
    rng = np.random.default_rng(seed)
    prob = (lam_per_min / 60.0) * dt
    events = rng.random(n) < prob
    for t0 in np.flatnonzero(events):
        t1 = min(n, t0 + h.size)
        p[t0:t1] += pulse_amp * h[: t1 - t0]
    return p

def make_single_ictal_p(
    T, dt,
    *, mu_base=145.0, sigma_base=20.0,
    t_onset=40.0, ictal_duration=30.0,
    ictal_height=120.0,            # moderate step (avoid fixed-point saturation)
    edge_rise_s=0.75, edge_fall_s=0.75,
    seed=7
):
    """
    Baseline OU(mu_base, sigma_base) → sharp logistic rise at t_onset to +ictal_height
    for ictal_duration → sharp fall back to baseline.
    Returns (p, t_on, t_off, mask).
    """
    n = int(np.round(T / dt))
    t = np.arange(n) * dt
    p = ou_noise(T, dt, mu=mu_base, tau=0.05, sigma=sigma_base, seed=seed)

    def logistic_edge(tt, t0, width):
        k = 6.0 / width
        return 1.0 / (1.0 + np.exp(-k * (tt - t0)))

    t_on = float(t_onset)
    t_off = t_on + float(ictal_duration)

    rise = logistic_edge(t, t_on, edge_rise_s)
    fall = logistic_edge(t, t_off, edge_fall_s)
    plateau = np.clip(rise - fall, 0.0, 1.0)

    p = p + ictal_height * plateau
    return np.clip(p, 0.0, None), t_on, t_off, plateau.astype(int)

def add_cluster_pulses_during_mask(p, dt, mask, *,
                                   rate_hz=3.5, pulse_amp=850.0,
                                   tau=0.010, seed=11):
    """
    Adds a Poisson train of alpha-shaped pulses only where mask==1 (ictal window).
    Creates grouped spikes during the ictal plateau.
    """
    rng = np.random.default_rng(seed)
    n = p.size
    h = alpha_pulse_kernel(dt, tau=tau, length=6 * tau)
    prob = rate_hz * dt
    events = (rng.random(n) < prob) & (mask.astype(bool))
    for t0 in np.flatnonzero(events):
        t1 = min(n, t0 + h.size)
        p[t0:t1] += pulse_amp * h[: t1 - t0]
    return p

# --------------------- baseline alignment (no filtering) ---------------------

def align_baselines(lfp, mask, fs, ramp_s=1.0):
    """
    Return a baseline-aligned version of 'lfp' where the mean during the ictal region
    matches the baseline mean, without any band-pass filtering.

    We subtract delta * r(t), where delta = mean(lfp[mask==1]) - mean(lfp[mask==0]),
    and r(t) is a smoothed (ramped) version of the 0/1 mask to avoid hard edges.
    """
    lfp = np.asarray(lfp, dtype=float)
    m = np.asarray(mask, dtype=float)
    if m.sum() == 0 or m.sum() == len(m):
        return lfp.copy()  # nothing to align

    mu_base = lfp[m == 0].mean()
    mu_ict  = lfp[m == 1].mean()
    delta = mu_ict - mu_base

    # Smooth the mask with a short moving-average window -> soft ramps at edges
    n_ramp = max(1, int(round(ramp_s * fs)))
    kernel = np.ones(n_ramp, dtype=float) / n_ramp
    r = np.convolve(m, kernel, mode="same")
    r = np.clip(r, 0.0, 1.0)  # guard tiny overshoots

    return lfp - delta * r

# --------------------- integrate()-based resolver ---------------------

MODEL_DEF = "model_templates.neural_mass_models.jansenrit.JRC"

# Candidate outputs (EPSP/IPSP pairs preferred; somatic potentials as fallback)
OUTPUT_CANDIDATES = [
    {"psp_e": "pc/rpo_e_in/v", "psp_i": "pc/rpo_i/v"},
    {"psp_e": "pc/rpo_e_in/V", "psp_i": "pc/rpo_i/V"},
    {"psp_e": "pc/rpo_e/v",    "psp_i": "pc/rpo_i/v"},
    {"psp_e": "pc/rpo_e/V",    "psp_i": "pc/rpo_i/V"},
    {"psp_e": "PC/RPO_e_in/v", "psp_i": "PC/RPO_i/v"},
    {"psp_e": "PC/RPO_e_in/V", "psp_i": "PC/RPO_i/V"},
    {"psp_e": "PC/RPO_e/v",    "psp_i": "PC/RPO_i/v"},
    {"psp_e": "PC/RPO_e/V",    "psp_i": "PC/RPO_i/V"},
    {"V": "pc/PRO/V"}, {"V": "PC/PRO/V"},
    {"V": "pc/pro/V"}, {"V": "PC/pro/V"},
    {"V": "pc/soma/V"},{"V": "PC/SOMA/V"},
    {"V": "pc/out/V"}, {"V": "PC/OUT/V"},
    {"V": "pc/PRO/v"}, {"V": "PC/PRO/v"},
]

# Candidate excitatory input hooks (for p(t))
INPUT_CANDIDATES = [
    "pc/rpo_e_in/m_in", "pc/rpo_e/m_in",
    "PC/RPO_e_in/m_in", "PC/RPO_e/m_in",
    "pc/rpo_e_in/u_in", "PC/RPO_e_in/u_in",
    "pc/PRO/m_in", "PC/PRO/m_in",
    "pc/PRO/u_in", "PC/PRO/u_in",
]

def try_integrate(sim_T, step_size, ds, outputs, p=None, input_var=None):
    kwargs = dict(
        simulation_time=sim_T,
        step_size=step_size,
        solver="scipy",
        sampling_step_size=ds,
        outputs=outputs,
        clear=True,
    )
    if p is not None and input_var is not None:
        kwargs["inputs"] = {input_var: p}
    return integrate(MODEL_DEF, **kwargs), outputs, input_var

def resolve_and_run(sim_T, fs, p):
    ds = 1.0 / fs
    step = ds
    last_err = None

    # 1) EPSP/IPSP with inputs
    for out in OUTPUT_CANDIDATES:
        if "psp_e" not in out:
            continue
        for iv in INPUT_CANDIDATES:
            try:
                res, o, iv_used = try_integrate(sim_T, step, ds, out, p, iv)
                print(f"[resolver] Using outputs={o}, input='{iv_used}'")
                return res, o, iv_used
            except Exception as e:
                last_err = e

    # 2) EPSP/IPSP without inputs
    for out in OUTPUT_CANDIDATES:
        if "psp_e" not in out:
            continue
        try:
            res, o, iv_used = try_integrate(sim_T, step, ds, out, None, None)
            print(f"[resolver] Using outputs={o}, input=None")
            return res, o, None
        except Exception as e:
            last_err = e

    # 3) Somatic V with inputs
    for out in OUTPUT_CANDIDATES:
        if "V" not in out:
            continue
        for iv in INPUT_CANDIDATES:
            try:
                res, o, iv_used = try_integrate(sim_T, step, ds, out, p, iv)
                print(f"[resolver] Using outputs={o}, input='{iv_used}'")
                return res, o, iv_used
            except Exception as e:
                last_err = e

    # 4) Somatic V without inputs
    for out in OUTPUT_CANDIDATES:
        if "V" not in out:
            continue
        try:
            res, o, iv_used = try_integrate(sim_T, step, ds, out, None, None)
            print(f"[resolver] Using outputs={o}, input=None")
            return res, o, None
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to resolve outputs/inputs via integrate(). Last error: {last_err}")

# --------------------- core simulator (raw + aligned) ---------------------

def run_jr(mode="interictal",
           duration=180.0,
           fs=1000.0,
           seed=7,
           save_dir="outputs"):
    """
    Simulate JR with either sparse interictal spikes or a single ictal epoch.
    NO filtering: returns & saves raw LFP, plus a baseline-aligned copy.
    """
    assert mode in ("interictal", "ictal")
    T = float(duration)
    ds = 1.0 / float(fs)
    transient = 5.0

    # Build drive and ictal mask
    if mode == "interictal":
        p = make_interictal_p(T, ds, mu=150.0, sigma=30.0,
                              lam_per_min=10.0, pulse_amp=800.0, tau=0.010,
                              seed=seed)
        mask = np.zeros(int(np.round(T / ds)), dtype=int)
        t_on = t_off = None
    else:
        p, t_on, t_off, mask = make_single_ictal_p(
            T, ds,
            mu_base=145.0, sigma_base=20.0,
            t_onset=40.0, ictal_duration=30.0,
            ictal_height=120.0,    # moderate step; avoid fixed-point lock
            edge_rise_s=0.75, edge_fall_s=0.75,
            seed=seed
        )
        # Add clustered pulses only during ictal to guarantee spiky epoch
        p = add_cluster_pulses_during_mask(
            p, ds, mask,
            rate_hz=3.5,
            pulse_amp=850.0,
            tau=0.010,
            seed=seed + 1
        )

    # Resolve outputs/inputs and run
    results, outputs_used, input_used = resolve_and_run(T, fs, p)

    # LFP proxy (raw)
    if "psp_e" in outputs_used:
        psp_e = results["psp_e"].to_numpy()
        psp_i = results["psp_i"].to_numpy()
        lfp = psp_e - psp_i
    else:
        lfp = results["V"].to_numpy()
    t = results.index.to_numpy()

    # Remove transient and align mask
    keep = t >= transient
    t, lfp = t[keep], lfp[keep]
    mask = mask[keep]

    # Baseline alignment (no filtering)
    lfp_aligned = align_baselines(lfp, mask=mask, fs=float(fs), ramp_s=1.0)

    # PSD on demeaned raw (for a quick spectral summary)
    spec_signal = lfp - np.mean(lfp)
    f, Pxx = welch(spec_signal, fs=float(fs), nperseg=4096, noverlap=2048)
    f_dom = float(f[np.argmax(Pxx)])

    # Save CSV (raw + aligned)
    outdir = Path(save_dir); outdir.mkdir(parents=True, exist_ok=True)
    stem = f"jr_lfp_{mode}"
    csv_path = outdir / f"{stem}.csv"
    fig_path = outdir / f"{stem}.png"

    df = pd.DataFrame({
        "t_s": t,
        "lfp_raw": lfp,
        "lfp_aligned": lfp_aligned,
        "ictal_mask": mask
    })
    df.to_csv(csv_path, index=False)

    # Plot raw (interictal) or baseline-aligned (ictal) with shading
    plt.figure(figsize=(12, 3.8))
    if mode == "ictal":
        series_to_plot = lfp_aligned
        on_idx = np.where(mask == 1)[0]
        if on_idx.size:
            t_start, t_end = t[on_idx[0]], t[on_idx[-1]]
            plt.axvspan(t_start, t_end, alpha=0.15)
            plt.text(t_start, np.max(series_to_plot) * 0.85, "ictal epoch", fontsize=10)
        title = "Single ictal epoch (baseline-aligned, RAW content)"
    else:
        series_to_plot = lfp
        title = "Interictal spikes (RAW)"
    plt.plot(t, series_to_plot, linewidth=0.9)
    plt.title(f"JR synthetic LFP — {title}")
    plt.xlabel("Time [s]"); plt.ylabel("LFP (a.u.)")
    plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()

    # Console amplitude summary on raw and aligned
    if mode == "ictal" and np.any(mask == 1):
        base_idx = mask == 0
        ict_idx  = mask == 1
        amp_base_raw = float(np.std(lfp[base_idx]))
        amp_ict_raw  = float(np.std(lfp[ict_idx]))
        amp_base_al  = float(np.std(lfp_aligned[base_idx]))
        amp_ict_al   = float(np.std(lfp_aligned[ict_idx]))
        print(f"[ictal] onset≈{t_on:.2f}s, offset≈{t_off:.2f}s")
        print(f"[ictal] std(raw) baseline={amp_base_raw:.3g}, ictal={amp_ict_raw:.3g}, "
              f"ratio={amp_ict_raw/max(amp_base_raw,1e-12):.2f}x")
        print(f"[ictal] std(aligned) baseline={amp_base_al:.3g}, ictal={amp_ict_al:.3g}, "
              f"ratio={amp_ict_al/max(amp_base_al,1e-12):.2f}x")

    print(f"[{mode}] Saved: {csv_path}")
    print(f"[{mode}] Saved: {fig_path}")
    print(f"[{mode}] Dominant frequency (raw, demeaned): {f_dom:.2f} Hz")
    print(f"[{mode}] Outputs used: {outputs_used} | Input used: {input_used}")

    meta = {"outputs": outputs_used, "input": input_used}
    return t, lfp, lfp_aligned, f_dom, meta

# --------------------- entry point ---------------------

if __name__ == "__main__":
    # Interictal spikes (180 s)
    run_jr(mode="interictal", duration=180.0, fs=1000.0, seed=7, save_dir="outputs")

    # Single ictal epoch with grouped spikes (180 s)
    run_jr(mode="ictal",      duration=180.0, fs=1000.0, seed=7, save_dir="outputs")
