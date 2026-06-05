"""
Validate the SBI pure-idle-spine probabilities (P0, P00, P000, …) against
the simulated state-probability vectors in merged_all_infinite_buffer.csv.

State indexing (from April16.pdf, Figure 1)
───────────────────────────────────────────
States are binary strings ordered by layer; within a layer, ascending binary.
The empty state ∧ is index 0. The pure-idle spine 0^(n) is the FIRST state of
each layer block, at index 2^n − 1:

    ∧ → 0,  "0" → 1,  "00" → 3,  "000" → 7,  "0000" → 15,  "00000" → 31

This mapping is verified below against the known analytical identities
(P∧ = e^(−λ/γ); P0+P1 = S; λP_x = |x|γ(P_x0+P_x1)).

SBI spine formula under test (Matrix.pdf, Response 2)
─────────────────────────────────────────────────────
    β₀(n) = γ / (µ + nγ + (λ/(n+1))(1 − β₀(n+1)))
    P0    = λ / (µ + γ + (λ/2)(1 − β₀(2))) · e^(−λ/γ)
    R_k   = λ / (kγ + λ(1 − β₀(k)))            (k ≥ 2)
    P_{0^(n)} = P0 · ∏_{k=2}^n R_k
"""

import re
import numpy as np
import pandas as pd

# ─── Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("/Users/sdatta/Downloads/Claude/merged_all_infinite_buffer.csv")
df = df[pd.to_numeric(df["lambda"], errors="coerce").notna()].copy()
for c in ("lambda", "mu", "gamma"):
    df[c] = df[c].astype(float)

def parse(cell):
    arrs = []
    for m in re.finditer(r"array\(\[([^\]]*)\]\)", cell):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))
        arrs.append([float(x) for x in nums])
    return np.array(arrs)

sp  = np.array([parse(c).mean(axis=0) for c in df["results_state_probabilities"]])
lam = df["lambda"].values
mu  = df["mu"].values
gam = df["gamma"].values
N   = len(df)

r2 = lambda y, yh: 1 - np.sum((y - yh) ** 2) / np.sum((y - np.mean(y)) ** 2)

# ─── 1. Verify the index mapping ─────────────────────────────────────────────
print("=" * 64)
print("1. Index-mapping verification (analytical identities)")
print("=" * 64)
Pw = np.exp(-lam / gam)
S  = (lam / gam) * Pw
print(f"  idx0 = P∧ = e^(−λ/γ)            R² = {r2(sp[:,0], Pw):.6f}")
print(f"  idx1+idx2 = S = (λ/γ)e^(−λ/γ)   R² = {r2(S, sp[:,1]+sp[:,2]):.6f}")
print(f"  λP0 = 2γ(P00+P01)  [1;3,4]      R² = {r2(sp[:,1], 2*gam*(sp[:,3]+sp[:,4])/lam):.6f}")
print(f"  λP00 = 3γ(P000+P001) [3;7,8]    R² = {r2(sp[:,3], 3*gam*(sp[:,7]+sp[:,8])/lam):.6f}")
print("  → spine indices confirmed: 1, 3, 7, 15, 31  (= 2^n − 1)")
print()

# ─── 2. SBI spine probabilities ──────────────────────────────────────────────
def beta_seq(l, m, g, M):
    b = np.zeros(M + 2)
    for n in range(M, 0, -1):
        b[n] = g / (m + n * g + (l / (n + 1)) * (1 - b[n + 1]))
    return b

idx_map = {1: 1, 3: 2, 7: 3, 15: 4, 31: 5}   # data index → spine depth n
P_sbi   = {k: np.zeros(N) for k in idx_map}
for i in range(N):
    l, m, g = lam[i], mu[i], gam[i]
    M = max(int(np.ceil(5 * l / g)) + 12, 40)
    b = beta_seq(l, m, g, M)
    P0 = np.exp(-l / g) * l / (m + g + (l / 2) * (1 - b[2]))
    P_sbi[1][i] = P0
    prod = P0
    depth_to_idx = {2: 3, 3: 7, 4: 15, 5: 31}
    for n in range(2, 6):
        prod *= l / (n * g + l * (1 - b[n]))
        P_sbi[depth_to_idx[n]][i] = prod

labels = {1: "0", 3: "00", 7: "000", 15: "0000", 31: "00000"}

print("=" * 64)
print("2. SBI spine probabilities vs simulated data")
print("=" * 64)
print(f"  {'state':>7} {'idx':>4}  {'R²(SBI)':>9}  {'data mean':>10}  {'SBI mean':>10}")
for idx in [1, 3, 7, 15, 31]:
    print(f"  {labels[idx]:>7} {idx:>4}  {r2(sp[:,idx], P_sbi[idx]):>9.4f}  "
          f"{sp[:,idx].mean():>10.6f}  {P_sbi[idx].mean():>10.6f}")
print()

# ─── 3. The swap: data P0 = S·µ/(µ+γ),  SBI P0 = S·γ/(µ+γ) ───────────────────
print("=" * 64)
print("3. Diagnosis: SBI inverts the idle/active split at P0")
print("=" * 64)
light = lam < mu
print(f"  Light-load subset (λ<µ, {light.sum()} rows):")
print(f"    data P0 vs S·µ/(µ+γ) :  R² = {r2(sp[light,1], S[light]*mu[light]/(mu[light]+gam[light])):.6f}  (idle = LARGE share)")
print(f"    SBI  P0 vs S·γ/(µ+γ) :  R² = {r2(P_sbi[1][light], S[light]*gam[light]/(mu[light]+gam[light])):.6f}  (SBI puts SMALL share on idle)")
print()
print("  Sample rows (λ=1.5, µ=50):")
print(f"    {'γ':>6}  {'data P0':>9}  {'SBI P0':>9}  {'S·µ/(µ+γ)':>10}  {'S·γ/(µ+γ)':>10}")
for i in range(5):
    print(f"    {gam[i]:6.2f}  {sp[i,1]:9.5f}  {P_sbi[1][i]:9.5f}  "
          f"{S[i]*mu[i]/(mu[i]+gam[i]):10.5f}  {S[i]*gam[i]/(mu[i]+gam[i]):10.5f}")
print()
print("  → SBI P0 ≈ S·γ/(µ+γ) is the ACTIVE-state (P1) probability, not P0.")
print("    The Response-2 derivation (p.8) sets P1/P0 ≈ µ/γ ≫ 1, but the data")
print("    requires P0/P1 = µ/γ (the validated ratio r = µ/γ). The spine is")
print("    therefore wrong at its anchor, and the loss rate built on it fails.")
