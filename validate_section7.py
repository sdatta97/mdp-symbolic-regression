"""
Validate the Matrix.pdf "Section 7 — Complete Response" derivation against
the simulated state probabilities and loss rates in merged_all_infinite_buffer.csv.

Section 7 consolidates Responses 4–6 into a self-contained pipeline:

  Layer 0/1:
    P∧ = e^(−λ/γ),   P0 + P1 = S ≡ (λ/γ)e^(−λ/γ)
    β₀(2) = γ / (µ + 2γ + λ/3)                      (Layer-3 truncation)
    K = µ(2γ+µ+λ) / [(2γ+µ)(γ + λβ₀(2))],  χ = K/(1+K)
    P0 = χ·S,   P1 = (1−χ)·S

  Layer 2:
    P11 = λ/(2γ+µ)·P1,   P10 = λµ/(2γ(2γ+µ))·P1
    P00 = (1−β₀(2))·(λ/2γ)·P0,   P01 = β₀(2)·(λ/2γ)·P0

  Spine / loss:
    R_n = (1−β₀(n))·λ/(nγ),   P_{0^(n)} = P0·∏_{k=2}^n R_k
    Loss Rate = λ − µ(1 − P∧ − Σ_{n≥1} P_{0^(n)})

State indexing (April16.pdf, Fig. 1):
    P∧=0, P0=1, P1=2, P00=3, P01=4, P10=5, P11=6, P000=7, P0000=15.
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

sp   = np.array([parse(c).mean(axis=0) for c in df["results_state_probabilities"]])
lam  = df["lambda"].values; mu = df["mu"].values; gam = df["gamma"].values
loss = pd.to_numeric(df["avg_packet_loss_rate"], errors="coerce").values
N    = len(df)
r2   = lambda y, yh, m: 1 - np.sum((y[m]-yh[m])**2) / np.sum((y[m]-np.mean(y[m]))**2)
allr = np.ones(N, bool)
S    = (lam/gam)*np.exp(-lam/gam); Pw = np.exp(-lam/gam)

def beta0_seq(l, m, g, M):
    b = np.zeros(M+2)
    for n in range(M, 0, -1):
        b[n] = g/(m + n*g + (l/(n+1))*(1 - b[n+1]))
    return b

P = {idx: np.zeros(N) for idx in (1, 3, 5, 6, 7, 15)}
spine_sum = np.zeros(N)
for i in range(N):
    l, m, g = lam[i], mu[i], gam[i]
    b2  = g/(m + 2*g + l/3)                         # Section 7 truncated β₀(2)
    K   = m*(2*g + m + l)/((2*g + m)*(g + l*b2))    # P0/P1
    chi = K/(1 + K)
    P0  = chi*S[i]; P1 = (1 - chi)*S[i]
    P[1][i] = P0
    P[6][i] = l/(2*g + m)*P1                        # P11
    P[5][i] = l*m/(2*g*(2*g + m))*P1                # P10
    P[3][i] = (1 - b2)*(l/(2*g))*P0                 # P00
    M = max(int(np.ceil(5*l/g)) + 2, 60); bb = beta0_seq(l, m, g, M)
    prod = P0; ssum = P0
    d2i = {2: 3, 3: 7, 4: 15}
    for n in range(2, M+1):
        prod *= (1 - bb[n])*l/(n*g)
        if n in d2i: P[d2i[n]][i] = prod
        ssum += prod
        if abs(prod) < 1e-15 and n > 5: break
    spine_sum[i] = ssum

print("=" * 60)
print("1. State probabilities (Section 7)")
print("=" * 60)
lab = {1: "P0", 6: "P11", 5: "P10", 3: "P00", 7: "P000", 15: "P0000"}
print(f"  {'state':>6} {'idx':>4}   {'R²(all)':>9}")
for idx in (1, 6, 5, 3, 7, 15):
    print(f"  {lab[idx]:>6} {idx:>4}   {r2(sp[:,idx], P[idx], allr):>9.4f}")
print()

print("=" * 60)
print("2. Loss rate (Section 7)  vs  avg_packet_loss_rate")
print("=" * 60)
loss_frac = (lam - mu*(1 - Pw - spine_sum))/lam
light = lam < mu; heavy = lam >= mu
print(f"  overall      R² = {r2(loss, loss_frac, allr):>8.4f}   (all prior responses: −147 to −32, negative)")
print(f"  light (λ<µ)  R² = {r2(loss, loss_frac, light):>8.4f}")
print(f"  heavy (λ≥µ)  R² = {r2(loss, loss_frac, heavy):>8.4f}")
print(f"  competing-exp γ/(µ+γ): R² = {r2(loss, gam/(mu+gam), allr):>8.4f}")
print()
print("  by µ:")
for mv in sorted(np.unique(mu)):
    mm = mu == mv
    print(f"    µ={int(mv):3d}: R²={r2(loss, loss_frac, mm):>8.4f}   meanErr={np.abs(loss[mm]-loss_frac[mm]).mean():.4f}")
print()
print("  Sample (λ=1.5, µ=50):")
print(f"    {'γ':>6}  {'sim loss':>9}  {'S7 loss':>9}")
for i in range(5):
    print(f"    {gam[i]:6.2f}  {loss[i]:9.4f}  {loss_frac[i]:9.4f}")
