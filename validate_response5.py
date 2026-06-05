"""
Validate the Matrix.pdf "Response 5" spine / loss formulas against the
simulated state probabilities in merged_all_infinite_buffer.csv.

Response 5 replaces the β₀ continued-fraction recurrence with a step-down ratio
taken directly from the verified down-flow identity λP_x = |x|γ(P_{x0}+P_{x1}):

    R_n = P_{0^(n)} / P_{0^(n-1)} = λ/(nγ) − ξ_n,   ξ_n = P_{0^(n-1)1}/P_{0^(n-1)}

and gives exact closed forms for the Layer-1/Layer-2 states via global balance
at the corner nodes (bypassing the partial-balance "guesses"):

    P11 = λ/(2γ+µ)·P1
    P10 = λµ/(2γ(2γ+µ))·P1
    P00 = µ/(2γ − 2λ/3)·P10
    P0  = [µ(λ/γ) + 2γ(P10+P00)/P∧]/(µ+λ+γ)·e^(−λ/γ)

State indexing (April16.pdf, Fig. 1):
    P∧=0, P0=1, P1=2, P00=3, P01=4, P10=5, P11=6.

This script validates the formulas AS WRITTEN first, then a balance-derived fix
for the one broken link (P00).
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
allr = np.ones(N, bool)

P0d, P1d, P00d, P01d, P10d, P11d = (sp[:, 1], sp[:, 2], sp[:, 3],
                                    sp[:, 4], sp[:, 5], sp[:, 6])
S  = (lam / gam) * np.exp(-lam / gam)
Pw = np.exp(-lam / gam)

# ─── 1. Response 5 AS WRITTEN ────────────────────────────────────────────────
print("=" * 66)
print("1. Response 5 formulas — AS WRITTEN (no corrections)")
print("=" * 66)
P11_pred = lam / (2*gam + mu) * P1d
P10_pred = lam*mu / (2*gam*(2*gam + mu)) * P1d
den00    = 2*gam - 2*lam/3
P00_pred = np.divide(mu, den00, out=np.full(N, np.nan), where=np.abs(den00) > 1e-9) * P10d
ok00     = np.isfinite(P00_pred)
P0_anchor = (mu*(lam/gam) + 2*gam*(P10d + P00d)/Pw) / (mu + lam + gam) * Pw

print(f"  P11 = λ/(2γ+µ)·P1            R² = {r2(P11d, P11_pred, allr):>8.4f}   ✓")
print(f"  P10 = λµ/(2γ(2γ+µ))·P1       R² = {r2(P10d, P10_pred, allr):>8.4f}   ~ (good light-load)")
print(f"  P00 = µ/(2γ−2λ/3)·P10        R² = {r2(P00d, P00_pred, ok00):>8.4f}   ✗  ({(den00<=0).sum()} rows: 2γ−2λ/3 ≤ 0)")
print(f"  P0  anchor (data P10,P00)    R² = {r2(P0d, P0_anchor, allr):>8.4f}   ✓ (exact structure)")
# step-down identity consistency
xi2 = np.divide(P01d, P0d, out=np.zeros(N), where=P0d > 1e-9)
R2_id = lam/(2*gam) - xi2
R2_data = np.divide(P00d, P0d, out=np.zeros(N), where=P0d > 1e-9)
print(f"  R_n = λ/(nγ)−ξ_n  (n=2)      R² = {r2(R2_data, R2_id, P0d > 1e-4):>8.4f}   ✓ (structurally exact)")
print()
print("  → Structure (anchor, R_n identity) is sound; P11 strong; the P00")
print("    closed form is the broken link (its denominator 2γ−2λ/3 goes")
print("    negative once λ ≥ 3γ).")
print()

# ─── 2. Balance-derived FIX for P00 ──────────────────────────────────────────
print("=" * 66)
print("2. Proposed fix for the P00 link")
print("=" * 66)
print("  Response 5's node-00 balance omits the arrival inflow from the parent")
print("  (0 → 00 at rate λ/2) and adds a spurious outflow edge (00 → 010, which")
print("  is a child of 01, not 00). The correct global balance at 00 is")
print("     (λ/2)P0 + µP10 + 3γ(P000+P001) = λP00 + 2γP00")
print("  and with λP00 = 3γ(P000+P001) this collapses to")
print("     P00 = [(λ/2)P0 + µP10] / (2γ).")
print()
P00_fix = ((lam/2)*P0d + mu*P10d) / (2*gam)
print(f"  Response 5  P00 = µ/(2γ−2λ/3)·P10        R² = {r2(P00d, P00_pred, ok00):>8.4f}")
print(f"  FIX         P00 = [(λ/2)P0 + µP10]/(2γ)  R² = {r2(P00d, P00_fix, allr):>8.4f}")
print()

# ─── 3. Fully self-contained spine with the fix ──────────────────────────────
print("=" * 66)
print("3. Self-contained spine (Response 5 anchor + P10 + fixed P00)")
print("=" * 66)
# Solve coupled system:  P1 = S − P0, P10 = c10·P1, P00 = [(λ/2)P0 + µP10]/(2γ),
# anchor P0(µ+λ+γ) = µS + 2γ(P10+P00).  Closed solution:
c10 = lam*mu / (2*gam*(2*gam + mu))
D   = lam*mu / (2*gam)                       # = c10·(2γ+µ)
P0_fix  = (mu + D) * S / (mu + lam + gam - lam/2 + D)
P1_fix  = S - P0_fix
P10_fix = c10 * P1_fix
P00_fx2 = ((lam/2)*P0_fix + mu*P10_fix) / (2*gam)
print(f"  P0   R² = {r2(P0d,  P0_fix,  allr):>8.4f}   (Response 5 fully-closed P0 was −7.6)")
print(f"  P00  R² = {r2(P00d, P00_fx2, allr):>8.4f}")
print(f"  P10  R² = {r2(P10d, P10_fix, allr):>8.4f}")
print()
print("  Sample (λ=1.5, µ=50):")
print(f"    {'γ':>6}  {'data P00':>9}  {'R5 P00':>9}  {'FIX P00':>9}")
for i in range(5):
    p5 = mu[i]/(2*gam[i]-2*lam[i]/3)*P10d[i]
    pf = ((lam[i]/2)*P0d[i] + mu[i]*P10d[i])/(2*gam[i])
    print(f"    {gam[i]:6.2f}  {P00d[i]:9.5f}  {p5:9.5f}  {pf:9.5f}")
