"""
Verify Doc11.docx's "exact match" claim against this repo's validation.

Doc11 tabulates 4 scenarios (µ = 0.5–2.5) with a ~1e-7 match between
"Empirical Simulation" and a "Pure Scalar Formula". This script shows:

  1. None of Doc11's parameter sets are in merged_all_infinite_buffer.csv
     (which has µ = 50–200) — a disjoint regime.
  2. The genuine Section 7 scalar formula (verified against the PDF's own worked
     example) does NOT reproduce Doc11's "Formula" column — it differs by 0.05–0.13.
  3. The 1e-7 agreement is the conservation tautology Σ_active ≡ 1 − P∧ − Σ_idle,
     which holds for ANY consistent state vector and tests only the partition,
     not the scalar formula.
"""

import re
import numpy as np
import pandas as pd

# ─── Section 7 scalar formula (verified vs PDF worked example) ───────────────
def beta0_seq(l, m, g, M):
    b = np.zeros(M + 2)
    for n in range(M, 0, -1):
        b[n] = g / (m + n*g + (l/(n+1))*(1 - b[n+1]))
    return b

def section7_lossrate(l, m, g):
    Pw = np.exp(-l/g); S = (l/g)*Pw
    b2 = g/(m + 2*g + l/3)
    K  = m*(2*g + m + l)/((2*g + m)*(g + l*b2)); chi = K/(1 + K)
    P0 = chi*S
    M = max(int(np.ceil(5*l/g)) + 2, 200); bb = beta0_seq(l, m, g, M)
    prod = P0; ssum = P0
    for n in range(2, M + 1):
        prod *= (1 - bb[n])*l/(n*g); ssum += prod
        if abs(prod) < 1e-18 and n > 5: break
    return l - m*(1 - Pw - ssum)

# Sanity: reproduce PDF worked example λ=1.5,µ=2,γ=1 → P0=0.2254, P1=0.1093
l, m, g = 1.5, 2.0, 1.0
b2 = g/(m + 2*g + l/3); K = m*(2*g+m+l)/((2*g+m)*(g+l*b2)); chi = K/(1+K)
S = (l/g)*np.exp(-l/g)
print(f"Sanity check (PDF p.24, λ=1.5,µ=2,γ=1): P0={chi*S:.6f} (PDF 0.225407), "
      f"P1={(1-chi)*S:.6f} (PDF 0.109288)")
print()

# ─── 1+2. Doc11 scenarios ─────────────────────────────────────────────────────
df = pd.read_csv("/Users/sdatta/Downloads/Claude/merged_all_infinite_buffer.csv")
df = df[pd.to_numeric(df["lambda"], errors="coerce").notna()].copy()
for c in ("lambda", "mu", "gamma"): df[c] = df[c].astype(float)
print(f"Validation dataset µ values: {sorted(df['mu'].unique())}")
print()

scen = [("1 High Load",   3.5, 0.5, 1.8, 3.25055104, 3.25055097),
        ("2 Balanced",    1.5, 1.5, 1.0, 0.78121045, 0.78121021),
        ("3 High Service",1.2, 2.0, 1.0, 0.65471481, 0.65471466),
        ("4 Low Load",    0.5, 2.5, 2.0, 0.11903340, 0.11903337)]
print(f"  {'scenario':16}{'λ,µ,γ':13}{'inCSV':>6}{'Doc11 form':>12}{'my scalar':>11}{'diff':>9}")
for name, l, m, g, sim, form in scen:
    incsv = ((df['lambda']==l)&(df['mu']==m)&(df['gamma']==g)).sum()
    mine = section7_lossrate(l, m, g)
    print(f"  {name:16}{f'{l},{m},{g}':13}{incsv:>6}{form:>12.6f}{mine:>11.6f}{abs(mine-form):>9.3f}")
print("  → Doc11 scenarios are NOT in the CSV; the genuine scalar formula")
print("    differs from Doc11's 'Formula' column by 0.05–0.13.")
print()

# ─── 3. The conservation tautology ───────────────────────────────────────────
def parse(cell):
    a = []
    for mm in re.finditer(r"array\(\[([^\]]*)\]\)", cell):
        a.append([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", mm.group(1))])
    return np.array(a)
sp = np.array([parse(c).mean(axis=0) for c in df["results_state_probabilities"]])
lam, mu, gam = df["lambda"].values, df["mu"].values, df["gamma"].values
states = ['']; n = 1
while len(states) < sp.shape[1]:
    for v in range(2**n):
        states.append(format(v, '0'+str(n)+'b'))
        if len(states) >= sp.shape[1]: break
    n += 1
busy    = [i for i, s in enumerate(states) if s.startswith('1')]
allidle = [i for i, s in enumerate(states) if s and s.startswith('0')]
spine   = [i for i, s in enumerate(states) if s and set(s) == {'0'}]
Pbusy = sp[:, busy].sum(1); Pall = sp[:, allidle].sum(1); Pspine = sp[:, spine].sum(1)
Pw = np.exp(-lam/gam)
i = 0
print("The conservation tautology (row λ=1.5,µ=50,γ=2.5):")
print(f"  λ − µ·Σ_active             = {lam[i]-mu[i]*Pbusy[i]:.6f}")
print(f"  λ − µ(1−P∧−Σ_ALL_idle)     = {lam[i]-mu[i]*(1-Pw[i]-Pall[i]):.6f}   ← ≈ identical (partition)")
print(f"  λ − µ(1−P∧−Σ_PURE_spine)   = {lam[i]-mu[i]*(1-Pw[i]-Pspine[i]):.6f}   ← the SCALAR formula")
print(f"  mean Σ_pure_spine={Pspine.mean():.4f} vs Σ_all_idle={Pall.mean():.4f}  (pure spine ≠ all idle)")
print()
print("CONCLUSION: Doc11's 1e-7 'match' uses Σ_all_idle / matrix values on both")
print("sides (a tautology). The pure-spine scalar formula gives a different answer.")
