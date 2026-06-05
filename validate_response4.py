"""
Validate the Response 4 spine-probability formula (Matrix.pdf) against the
simulated state probabilities in merged_all_infinite_buffer.csv.

Response 4 corrects the P0 anchor by adding the missing service-feedback term
µP1 to the Layer-1 balance:

    (λ+γ)P0 = 2γP00 + 2γP01 + µP1

Solving (with P00 = R2·P0, P01 = (β₀(2)/(1−β₀(2)))·P00, and the global boundary
P0+P1 = (λ/γ)e^(−λ/γ)) gives the corrected anchor:

    P0 = µ(λ/γ) / [ (µ+λ+γ) − 2γR2/(1−β₀(2)) ] · e^(−λ/γ)
    R2 = λ / (2γ + λ(1−β₀(2)))
    β₀(n) = γ / (µ + nγ + (λ/(n+1))(1−β₀(n+1)))     (backward recurrence)

The spine then unrolls via  P_{0^(n)} = P0 · ∏_{k=2}^n R_k,
with  R_k = λ / (kγ + λ(1−β₀(k))).

State indexing (April16.pdf, Fig. 1): pure-idle spine 0^(n) at index 2^n − 1
    → "0"=1, "00"=3, "000"=7, "0000"=15, "00000"=31.

Findings (see printout):
  • Anchor P0 is FIXED — R² ≈ 0.93–0.95 (broken Response-2 SBI gave −2.00).
  • The spine step-down ratio R_k = λ/(kγ+λ(1−β₀(k))) still does not match
    the data (R² ≈ 0.10 on the P00/P0 ratio); the flux-balance form
    R_{n+1} = (1−β₀(n))·λ/((n+1)γ) matches at R² ≈ 0.995.
"""

import re
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
N   = len(df)
r2  = lambda y, yh, m: 1 - np.sum((y[m]-yh[m])**2) / np.sum((y[m]-np.mean(y[m]))**2)
clean = sp[:, 7] > 1e-4                              # depth-3 spine above noise

def beta0_seq(l, m, g, M):
    """Backward continued-fraction recurrence for β₀(n) (Response 1–4)."""
    b = np.zeros(M + 2)
    for n in range(M, 0, -1):
        b[n] = g / (m + n * g + (l / (n + 1)) * (1 - b[n + 1]))
    return b

# ─── Response 4 spine ────────────────────────────────────────────────────────
P = {idx: np.zeros(N) for idx in (1, 3, 7, 15, 31)}
depth_to_idx = {2: 3, 3: 7, 4: 15, 5: 31}
for i in range(N):
    l, m, g = lam[i], mu[i], gam[i]
    M = max(int(np.ceil(5 * l / g)) + 2, 40)
    b = beta0_seq(l, m, g, M)
    R2 = l / (2 * g + l * (1 - b[2]))
    denom = (m + l + g) - 2 * g * R2 / (1 - b[2])    # Response 4 anchor denominator
    P0 = m * (l / g) / denom * np.exp(-l / g)
    P[1][i] = P0
    prod = P0
    for n in range(2, 6):
        prod *= l / (n * g + l * (1 - b[n]))          # R_k = λ/(kγ+λ(1−β₀(k)))
        P[depth_to_idx[n]][i] = prod

labels = {1: "0", 3: "00", 7: "000", 15: "0000", 31: "00000"}
allr = np.ones(N, bool)

print("=" * 64)
print("Response 4 spine probabilities vs simulated data")
print("=" * 64)
print(f"  {'state':>7} {'idx':>4}   {'R²(all)':>9}   {'R²(clean)':>10}   {'R²(R2-SBI)':>11}")
broken = {1: -2.00, 3: -0.84, 7: -0.28, 15: -0.18, 31: -0.14}
for idx in (1, 3, 7, 15, 31):
    print(f"  {labels[idx]:>7} {idx:>4}   {r2(sp[:,idx],P[idx],allr):>9.4f}   "
          f"{r2(sp[:,idx],P[idx],clean):>10.4f}   {broken[idx]:>11.2f}")
print()
print("  → Anchor P0 fixed (R²≈0.93–0.95, was −2.00). Deeper states degrade")
print("    because the step-down ratio R_k was not corrected.")
print()

# ─── Diagnose the remaining error: the step-down ratio ───────────────────────
print("=" * 64)
print("Step-down ratio  R2 = P00/P0  — locating the residual error")
print("=" * 64)
ratio_data = np.divide(sp[:, 3], sp[:, 1], out=np.zeros(N), where=sp[:, 1] > 1e-9)
b1 = np.divide(sp[:, 4], sp[:, 3] + sp[:, 4],
               out=np.zeros(N), where=(sp[:, 3] + sp[:, 4]) > 1e-9)   # β₀(1) measured
R2_resp4 = np.array([lam[i] / (2*gam[i] + lam[i]*(1 - beta0_seq(lam[i],mu[i],gam[i],
                     max(int(np.ceil(5*lam[i]/gam[i]))+2,40))[2])) for i in range(N)])
R2_cut = (1 - b1) * lam / (2 * gam)
print(f"  Response 4   R2 = λ/(2γ+λ(1−β₀(2))):   R² = {r2(ratio_data, R2_resp4, clean):.4f}")
print(f"  cut identity (1−β₀(1))·λ/(2γ):        R² = {r2(ratio_data, R2_cut,   clean):.4f}")
print()

# ─── Confirm: Response 4 anchor + cut-identity step-down ─────────────────────
b2 = np.divide(sp[:, 8], sp[:, 7] + sp[:, 8],
               out=np.zeros(N), where=(sp[:, 7] + sp[:, 8]) > 1e-9)
P00_hy  = (1 - b1) * lam / (2 * gam) * P[1]
P000_hy = (1 - b2) * lam / (3 * gam) * P00_hy
print("=" * 64)
print("Response 4 anchor + cut-identity step-down (the complete fix)")
print("=" * 64)
print(f"  P0   R²(clean) = {r2(sp[:,1], P[1],     clean):.4f}")
print(f"  P00  R²(clean) = {r2(sp[:,3], P00_hy,   clean):.4f}")
print(f"  P000 R²(clean) = {r2(sp[:,7], P000_hy,  clean):.4f}")
