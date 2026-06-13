#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-compile Gisola's Numba kernels and write them to the on-disk cache.

Run this script ONCE after installation (or after upgrading Gisola / Numba):

    cd src
    python precompile.py

All subsequent runs — including the very first production run — will load
the compiled kernels from disk instead of recompiling.

The cache lives in  ~/.numba_cache  (controlled by NUMBA_CACHE_DIR in isola2.py).
Re-run this script after upgrading Numba, Python, or any @njit function.
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import isola2 so NUMBA_CACHE_DIR is set before numba imports anything
import isola2

rng = np.random.default_rng(0)
NSTATIONS, WIN = 2, 1024

G_s       = rng.random((4, WIN * NSTATIONS, 5), dtype=np.float64)
Dobs_s    = rng.random(WIN * NSTATIONS, dtype=np.float64)
shifts_s  = np.array([0], dtype=np.int64)
W_s       = rng.random(WIN * NSTATIONS, dtype=np.float64)
Coords_s  = rng.random((4, 3), dtype=np.float64)

print("Pre-compiling Gisola Numba kernels ...")
t0 = time.time()

# 1. Inversion kernel
print("  [1/3] inversion kernel ...")
shift, mech, best_idx, vr, corr, svals, cond, vrm = isola2.compute_inversion_numbaCpu(
    G_s, Dobs_s, 0.1, shifts_s, NSTATIONS)

# 2. Focal-mechanism helpers
print("  [2/3] focal mechanism helpers ...")
_ = isola2.silsub(mech)

# 3. Bootstrap kernel
print("  [3/3] bootstrap kernel ...")
_ = isola2.compute_bootstraping_numbaCpu(
    G_s, Dobs_s, W_s, 0.1, shifts_s, Coords_s)

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f} s  —  zero JIT overhead on all future runs.")
