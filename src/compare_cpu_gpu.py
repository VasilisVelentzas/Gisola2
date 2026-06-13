#!/usr/bin/env python3
"""
Compare CPU vs GPU inversion and bootstrap results on synthetic data.
Run from GisolaBootstrap/src/:
    python compare_cpu_gpu.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import isola2

# ── reproducible synthetic data ──────────────────────────────────────────────
rng        = np.random.default_rng(42)
N_SOURCES  = 528          # bootstrap grid
N_SOURCES_INV = 3500      # full inversion grid
N_STATIONS = 4
N_SAMPLES  = 1024
N_TOTAL    = N_STATIONS * N_SAMPLES
N_COMP     = 5
N_ITERS    = 100
DT         = 0.4
SHIFTS     = np.arange(-10, 44, dtype=np.int64)   # 54 shifts

def make_data(n_sources):
    G    = rng.standard_normal((n_sources, N_TOTAL, N_COMP)).astype(np.float64)
    Dobs = rng.standard_normal(N_TOTAL).astype(np.float64)
    return G, Dobs

# ── warmup GPU (compile kernels) ─────────────────────────────────────────────
print("Warming up GPU kernels ...", flush=True)
isola2._USE_GPU = True
try:
    from numba import cuda as _cuda
    isola2._CUDA_AVAILABLE = len(list(_cuda.gpus)) > 0
except Exception:
    isola2._CUDA_AVAILABLE = False

if not isola2._CUDA_AVAILABLE:
    print("No CUDA GPU found — cannot compare.")
    sys.exit(1)

_G_wu  = rng.random((4, N_TOTAL, N_COMP))
_ob_wu = rng.random(N_TOTAL)
_sh_wu = np.array([0], dtype=np.int64)
_W_wu  = rng.random((1, N_TOTAL))
_Co_wu = rng.random((4, 3))
isola2.compute_inversion_numbaGpu(_G_wu, _ob_wu, DT, _sh_wu, 4)
isola2._bootstrap_gpu_loop(_G_wu, _ob_wu, _W_wu, DT, _sh_wu, _Co_wu)
print("GPU ready.\n", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. INVERSION comparison
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("INVERSION  ({} sources, {} shifts)".format(N_SOURCES_INV, len(SHIFTS)))
print("=" * 60)

G_inv, Dobs_inv = make_data(N_SOURCES_INV)

cpu_inv = isola2.compute_inversion_numbaCpu(G_inv, Dobs_inv, DT, SHIFTS, N_STATIONS)
gpu_inv = isola2.compute_inversion_numbaGpu(G_inv, Dobs_inv, DT, SHIFTS, N_STATIONS)

# unpack: best_shift, best_mech, best_idx, best_vr, best_corr, svals, cond, vr_mat
cpu_best_idx  = cpu_inv[2]
gpu_best_idx  = gpu_inv[2]
cpu_best_vr   = cpu_inv[3][cpu_best_idx]
gpu_best_vr   = gpu_inv[3][gpu_best_idx]
cpu_best_mech = cpu_inv[1][cpu_best_idx]
gpu_best_mech = gpu_inv[1][gpu_best_idx]

print("Best grid point  — CPU: {:4d}   GPU: {:4d}  match: {}".format(
    cpu_best_idx, gpu_best_idx, cpu_best_idx == gpu_best_idx))
print("Best VR          — CPU: {:.6f}   GPU: {:.6f}  diff: {:.2e}".format(
    cpu_best_vr, gpu_best_vr, abs(cpu_best_vr - gpu_best_vr)))
print("Best mech diff   — max |CPU-GPU|: {:.2e}".format(
    np.max(np.abs(cpu_best_mech - gpu_best_mech))))

# full VR array comparison
vr_diff = np.abs(cpu_inv[3] - gpu_inv[3])
print("All-source VR    — max diff: {:.2e}   mean diff: {:.2e}".format(
    vr_diff.max(), vr_diff.mean()))

# full mech array
mech_diff = np.abs(cpu_inv[1] - gpu_inv[1])
print("All-source mech  — max diff: {:.2e}   mean diff: {:.2e}".format(
    mech_diff.max(), mech_diff.mean()))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BOOTSTRAP comparison
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("BOOTSTRAP  ({} sources, {} iters, {} shifts)".format(
    N_SOURCES, N_ITERS, len(SHIFTS)))
print("=" * 60)

G_boot, Dobs_boot = make_data(N_SOURCES)
W     = rng.random((N_ITERS, N_TOTAL)).astype(np.float64)
Coords = rng.random((N_SOURCES, 3))

# CPU: iterate
cpu_boot = []
for i in range(N_ITERS):
    res = isola2.compute_bootstraping_numbaCpu(
        G_boot, Dobs_boot, W[i], DT, SHIFTS, Coords)
    cpu_boot.append(res)

# GPU: batched
gpu_boot = isola2._bootstrap_gpu_loop(
    G_boot, Dobs_boot, W, DT, SHIFTS, Coords)

# compare per iteration: best_vr, best_shift, best_mech of top-1
vr_diffs    = []
shift_match = []
mech_diffs  = []

for i in range(N_ITERS):
    # CPU top-1
    cpu_vr   = cpu_boot[i][3]
    cpu_idx0 = np.argmax(cpu_vr)           # best within kept subset
    cpu_best_vr_i   = cpu_vr[cpu_idx0]
    cpu_best_mech_i = cpu_boot[i][2][cpu_idx0]
    cpu_best_sh_i   = cpu_boot[i][0][cpu_idx0]

    # GPU top-1
    gpu_vr   = gpu_boot[i][3]
    gpu_idx0 = np.argmax(gpu_vr)
    gpu_best_vr_i   = gpu_vr[gpu_idx0]
    gpu_best_mech_i = gpu_boot[i][2][gpu_idx0]
    gpu_best_sh_i   = gpu_boot[i][0][gpu_idx0]

    vr_diffs.append(abs(cpu_best_vr_i - gpu_best_vr_i))
    shift_match.append(cpu_best_sh_i == gpu_best_sh_i)
    mech_diffs.append(np.max(np.abs(cpu_best_mech_i - gpu_best_mech_i)))

print("Best VR diff (per iter) — max: {:.2e}   mean: {:.2e}".format(
    max(vr_diffs), sum(vr_diffs)/len(vr_diffs)))
print("Best shift match        — {}/{} iters agree".format(
    sum(shift_match), N_ITERS))
print("Best mech diff (per iter)— max: {:.2e}   mean: {:.2e}".format(
    max(mech_diffs), sum(mech_diffs)/len(mech_diffs)))

print("\nDone.")
