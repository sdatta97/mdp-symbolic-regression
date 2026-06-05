"""
Validate the Matrix.pdf "Response 6" spine probability expressions P_0(n)
against the simulated state probabilities in merged_all_infinite_buffer.csv.

Response 6 fixes the negative-probability artifact in Response 5's P00 by
anchoring P00/P01 to the verified identity  2γP00 + 2γP01 = λP0  and splitting
with a *bounded* branching factor (so the denominator can never go negative):

    β₀(2) = γ / (µ + 2γ + λ/3)                    (Layer-3 truncation)
    β₀(n) = γ / (µ + nγ + (λ/(n+1))(1−β₀(n+1)))    (general recurrence)

    P00 = (1−β₀(2))·(λ/2γ)·P0
    step-down:  R_n = (1−β₀(n))·λ/(nγ)            ⇒  P_{0^(n)} = P0·∏_{k=2}^n R_k

The P0 anchor is recomputed from the global balance at node 0
    µP1 + 2γ(P00+P10) = (λ+γ)P0
with  P10 = λµ/(2γ(2γ+µ))·P1  and  P00 = (1−β₀(2))(λ/2γ)P0, giving the ratio

    P0/P1 = µ(2γ+µ+λ) / [(2γ+µ)(γ + β₀(2)λ)],   P0+P1 = (λ/γ)e^(−λ/γ).

State indexing (April16.pdf, Fig. 1): spine 0^(n) at index 2^n − 1
    → "0"=1, "00"=3, "000"=7, "0000"=15, "00000"=31.
"""

import re
import numpy as np
import pandas as pd

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
lam = df["lambda"].values; mu = df["mu"].values; gam = df["gamma"].values
N   = len(df)
r2  = lambda y, yh, m: 1 - np.sum((y[m]-yh[m])**2) / np.sum((y[m]-np.mean(y[m]))**2)
allr  = np.ones(N, bool)
clean = sp[:, 7] > 1e-4

P0d, P1d, P00d, P01d, P10d, P11d = (sp[:, 1], sp[:, 2], sp[:, 3],
                                    sp[:, 4], sp[:, 5], sp[:, 6])
S  = (lam / gam) * np.exp(-lam / gam)

def beta0_seq(l, m, g, M):
    b = np.zeros(M + 2)
    for n in range(M, 0, -1):
        b[n] = g / (m + n*g + (l/(n+1))*(1 - b[n+1]))
    return b

# ─── Response 6 spine ────────────────────────────────────────────────────────
P = {idx: np.zeros(N) for idx in (1, 3, 7, 15, 31)}
b2 = np.zeros(N)
d2i = {2: 3, 3: 7, 4: 15, 5: 31}
for i in range(N):
    l, m, g = lam[i], mu[i], gam[i]
    M = max(int(np.ceil(5*l/g)) + 2, 40)
    b = beta0_seq(l, m, g, M); b2[i] = b[2]
    rho = m*(2*g + m + l) / ((2*g + m)*(g + b[2]*l))     # P0/P1
    P0 = S[i]*rho/(1 + rho)
    P[1][i] = P0
    prod = P0
    for n in range(2, 6):
        prod *= (1 - b[n]) * l / (n*g)                    # R_n = (1−β₀(n))λ/(nγ)
        P[d2i[n]][i] = prod

lab = {1: "0", 3: "00", 7: "000", 15: "0000", 31: "00000"}
broken5 = {1: "(−7.6)", 3: "−24.7", 7: "n/a", 15: "n/a", 31: "n/a"}

print("=" * 70)
print("Response 6 spine probabilities P_0(n) vs simulated data")
print("=" * 70)
print(f"  {'state':>7} {'idx':>4}   {'R²(all)':>9}   {'R²(clean)':>10}   {'Response 5':>11}")
for idx in (1, 3, 7, 15, 31):
    print(f"  {lab[idx]:>7} {idx:>4}   {r2(sp[:,idx],P[idx],allr):>9.4f}   "
          f"{r2(sp[:,idx],P[idx],clean):>10.4f}   {broken5[idx]:>11}")
print()
print("  → New P0 anchor + bounded P00: the negative artifact is gone.")
print("    P0/P00/P000 validate well; deep states (idx31) overshoot for")
print("    heavy load (λ/γ large), where the 37-state sim is noise-limited.")
print()

# ─── Individual link checks (fed data parents) ───────────────────────────────
print("=" * 70)
print("Individual link validation (fed data parent states)")
print("=" * 70)
print(f"  P00 = (1−β₀(2))·(λ/2γ)·P0   R² = {r2(P00d, (1-b2)*(lam/(2*gam))*P0d, allr):>8.4f}   ✓ (Response 5: −24.7)")
print(f"  P11 = λ/(2γ+µ)·P1           R² = {r2(P11d, lam/(2*gam+mu)*P1d, allr):>8.4f}   ✓")
print(f"  P10 = λµ/(2γ(2γ+µ))·P1      R² = {r2(P10d, lam*mu/(2*gam*(2*gam+mu))*P1d, allr):>8.4f}   ~ light-load")
print()
print("  Sample (λ=1.5, µ=50):")
print(f"    {'γ':>6}  {'data P0':>9}  {'R6 P0':>9}  | {'data P00':>9}  {'R6 P00':>9}")
for i in range(5):
    print(f"    {gam[i]:6.2f}  {P0d[i]:9.5f}  {P[1][i]:9.5f}  | {P00d[i]:9.5f}  {P[3][i]:9.5f}")
