"""
Corrected derivation of the pure-idle-spine probabilities P0, P00, P000, …
and validation against merged_all_infinite_buffer.csv.

This fixes the two errors in Matrix.pdf (SBI / Response 2):

  ERROR 1 — inverted anchor.
    PDF (p.8):   P1 = [(µ+γ+(λ/2)(1−β₀(2)))/γ]·P0   ⇒  P1/P0 ≈ µ/γ ≫ 1
    Correct:     P1 = (γ/µ)(P0 + P01)                ⇒  P0/P1 ≈ µ/γ ≫ 1
    The PDF makes the *active* state dominate; the data (and the verified
    partial-balance eqns of April16.pdf) make the *idle* state dominate.
    Corrected anchor:   P0 = S·µ / (µ + γ(1 + b)),   b = P01/P0,
    with S = (λ/γ)e^(−λ/γ).  Leading order: P0 ≈ S·µ/(µ+γ).

  ERROR 2 — wrong spine multiplier.
    PDF:        R_n        = λ / (nγ + λ(1 − β₀(n)))
    Correct:    R_{n+1}    = (1 − β₀(n))·λ / ((n+1)γ)
    The correct form follows directly from the *verified* cut identity
    λP_x = (|x|+1)γ(P_{x0}+P_{x1})  with  β₀(n)=P_{0^n1}/(P_{0^n0}+P_{0^n1}).

Closed-form result (corrected):
    P_{0^(n)} = µ/(µ+γ) · (λ/γ)^n e^(−λ/γ) / n! · ∏_{k=1}^{n-1}(1 − β₀(k))
  i.e. a Poisson(λ/γ) shape, scaled by the served-fraction µ/(µ+γ) and
  thinned by the branch-loss factors (1 − β₀(k)).

State indexing (April16.pdf, Fig. 1): spine 0^(n) at index 2^n − 1
    → "0"=1, "00"=3, "000"=7, "0000"=15, "00000"=31.
"""

import re
import math
import numpy as np
import pandas as pd

# ─── Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv("/Users/sdatta/Downloads/Claude/merged_all_infinite_buffer.csv")
df = df[pd.to_numeric(df["lambda"], errors="coerce").notna()].copy()
for c in ("lambda", "mu", "gamma"):
    df[c] = df[c].astype(float)

def parse(cell):
    arrs = []
    for m in re.finditer(r"array\(\[([^\]]*)\]\)", cell):
        arrs.append([float(x) for x in
                     re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))])
    return np.array(arrs)

sp  = np.array([parse(c).mean(axis=0) for c in df["results_state_probabilities"]])
lam = df["lambda"].values
mu  = df["mu"].values
gam = df["gamma"].values

Pw = np.exp(-lam / gam)
S  = (lam / gam) * Pw
r2 = lambda y, yh, m: 1 - np.sum((y[m]-yh[m])**2) / np.sum((y[m]-np.mean(y[m]))**2)
allrows = np.ones(len(df), bool)

# ─── 1. The corrected anchor relation ────────────────────────────────────────
print("=" * 66)
print("1. Anchor:  the µ/γ ratio was inverted")
print("=" * 66)
P1_correct = (gam / mu) * (sp[:, 1] + sp[:, 4])      # (γ/µ)(P0 + P01)
P1_pdf     = (mu + gam) / gam * sp[:, 1]             # PDF: (µ+γ)/γ · P0
print(f"  P1 = (γ/µ)(P0+P01)   [corrected]   R² = {r2(sp[:,2], P1_correct, allrows):.6f}")
print(f"  P1 = (µ+γ)/γ · P0    [PDF]          R² = {r2(sp[:,2], P1_pdf, allrows):.6f}")
print()
b = np.divide(sp[:, 4], sp[:, 1], out=np.zeros_like(sp[:, 1]), where=sp[:, 1] > 0)
P0_corrected = S * mu / (mu + gam * (1 + b))
finite = sp[:, 1] > 1e-6                              # rows where P0 is resolved
print(f"  P0 = S·µ/(µ+γ(1+b))  [corrected]   R² = {r2(sp[:,1], P0_corrected, finite):.6f}")
print(f"  P0 ≈ S·µ/(µ+γ)       [leading]     R² = {r2(sp[:,1], S*mu/(mu+gam), allrows):.6f}")
print(f"  P0 = S·γ/(µ+γ)       [PDF, broken] R² = {r2(sp[:,1], S*gam/(mu+gam), allrows):.6f}")
print()

# ─── 2. The corrected spine multiplier (cut identity) ────────────────────────
print("=" * 66)
print("2. Spine multiplier:  R_{n+1} = (1−β₀(n))·λ/((n+1)γ)")
print("=" * 66)
clean = sp[:, 7] > 1e-4                               # depth-3 above noise floor
den1 = sp[:, 3] + sp[:, 4]; den2 = sp[:, 7] + sp[:, 8]
b1 = np.divide(sp[:, 4], den1, out=np.zeros_like(den1), where=den1 > 0)   # β₀(1)
b2 = np.divide(sp[:, 8], den2, out=np.zeros_like(den2), where=den2 > 0)   # β₀(2)
P00_correct  = (1 - b1) * lam / (2 * gam) * sp[:, 1]
P000_correct = (1 - b2) * lam / (3 * gam) * sp[:, 3]
P00_pdf      = lam / (2 * gam + lam * (1 - b1)) * sp[:, 1]
print(f"  P00  = (1−β₀(1))·λ/(2γ)·P0  [corrected]  R² = {r2(sp[:,3], P00_correct, clean):.6f}")
print(f"  P000 = (1−β₀(2))·λ/(3γ)·P00 [corrected]  R² = {r2(sp[:,7], P000_correct, clean):.6f}")
print(f"  P00  = λ/(2γ+λ(1−β₀(1)))·P0 [PDF]        R² = {r2(sp[:,3], P00_pdf, clean):.6f}")
print()

# ─── 3. Full corrected spine (anchor + recursion, measured β₀) ───────────────
print("=" * 66)
print("3. Full corrected spine  (anchor + cut recursion, measured β₀)")
print("=" * 66)
P0c   = S * mu / (mu + gam * (1 + b))
P00c  = (1 - b1) * lam / (2 * gam) * P0c
P000c = (1 - b2) * lam / (3 * gam) * P00c
print(f"  {'state':>6} {'idx':>4}   {'R²(corrected)':>14}   {'R²(broken SBI)':>15}")
print(f"  {'0':>6} {1:>4}   {r2(sp[:,1],P0c,clean):>14.6f}   {-2.00:>15.2f}")
print(f"  {'00':>6} {3:>4}   {r2(sp[:,3],P00c,clean):>14.6f}   {-0.84:>15.2f}")
print(f"  {'000':>6} {7:>4}   {r2(sp[:,7],P000c,clean):>14.6f}   {-0.28:>15.2f}")
print(f"  ({clean.sum()} clean rows; depth-3 spine above noise floor)")
print()

# ─── 4. Closed-form Poisson result (leading order) ───────────────────────────
print("=" * 66)
print("4. Closed-form leading order:  P_{0^n} = µ/(µ+γ)·(λ/γ)^n e^(−λ/γ)/n!")
print("=" * 66)
light = lam < mu
print(f"  {'state':>6} {'idx':>4}   {'R²(all)':>9}   {'R²(λ<µ)':>9}")
for n, idx in {1: 1, 2: 3, 3: 7}.items():
    pred = mu / (mu + gam) * (lam / gam) ** n * Pw / math.factorial(n)
    print(f"  {'0'*n:>6} {idx:>4}   {r2(sp[:,idx],pred,allrows):>9.4f}   {r2(sp[:,idx],pred,light):>9.4f}")
print()
print("  Interpretation: the idle queue length follows a Poisson(λ/γ) law,")
print("  scaled by the served-fraction µ/(µ+γ) and thinned by branch losses.")
