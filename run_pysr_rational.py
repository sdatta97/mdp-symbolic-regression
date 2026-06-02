"""
PySR — fit  r = p₁/p₂  in the restricted form  r = μ/(γ + f(λ,μ,γ))
where f is a polynomial fraction (rational function) only.

Motivation
──────────
  Previous unrestricted search already converged on this skeleton.
  Fixing the numerator as μ and fitting only f = μ/r − γ:
    • reduces the search space dramatically
    • guarantees the clean physics form r = μ/(γ + f)
    • polynomial-fraction restriction keeps results analytically tractable

  Once f is known:
      r  = μ/(γ + f)
      p₁ = S · r/(1+r) = S·μ/(μ + γ + f)
      p₂ = S · 1/(1+r) = S·(γ+f)/(μ + γ + f)
      p₁ + p₂ = S  ← exact by construction

Operators
─────────
  Binary : +  −  *  /                 (rational arithmetic only)
  Unary  : square                     (x² is polynomial; no exp/log/sqrt)

This restricts the search to rational functions P(λ,μ,γ)/Q(λ,μ,γ).

Dataset : merged_may21_all.csv  (5526 rows, μ ∈ {50,100,125,150,175,200})
Filter  : S > 0.001  AND  y₁ > 0.001  AND  y₂ > 0.001
"""

import re as _re
import numpy as np
import pandas as pd
from pysr import PySRRegressor

# ─── 1. Load & clean ─────────────────────────────────────────────────────────
df = pd.read_csv("/Users/sdatta/Downloads/Claude/may21/merged_may21_all.csv")
df = df[pd.to_numeric(df["lambda"], errors="coerce").notna()].copy()
for col in ("lambda", "mu", "gamma"):
    df[col] = df[col].astype(float)

print(f"Total rows : {len(df)}")

# ─── 2. Parse state probabilities ────────────────────────────────────────────
def parse_run_arrays(cell: str) -> np.ndarray:
    arrays = []
    for m in _re.finditer(r"array\(\[([^\]]*)\]\)", cell):
        nums = _re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))
        arrays.append([float(x) for x in nums])
    return np.array(arrays)

state_probs = np.array([
    parse_run_arrays(c).mean(axis=0)
    for c in df["results_state_probabilities"]
])

lam_all = df["lambda"].values
mu_all  = df["mu"].values
gam_all = df["gamma"].values
y1_all  = state_probs[:, 1]
y2_all  = state_probs[:, 2]
S_all   = (lam_all / gam_all) * np.exp(-lam_all / gam_all)

# ─── 3. Filter ───────────────────────────────────────────────────────────────
SMIN  = 1e-3
YMIN  = 1e-3    # need reliable y₁ and y₂ to form a good f target
mask  = (S_all > SMIN) & (y1_all > YMIN) & (y2_all > YMIN)

X    = np.column_stack([lam_all[mask], mu_all[mask], gam_all[mask]])
S    = S_all[mask]
y1   = y1_all[mask]
y2   = y2_all[mask]
lam  = X[:, 0];  mu = X[:, 1];  gam = X[:, 2]

# ─── 4. Compute target  f = μ/r − γ ─────────────────────────────────────────
r_data    = y1 / y2               # empirical ratio
f_data    = mu / r_data - gam     # f such that r = μ/(γ+f)

print(f"Training rows : {mask.sum()} / {len(df)}")
print(f"μ breakdown:")
for mu_v in sorted(np.unique(mu)):
    print(f"  μ={mu_v:.0f} : {(mu == mu_v).sum()} rows")

print(f"\nf = μ/r − γ  statistics:")
print(f"  min  = {f_data.min():.4f}")
print(f"  max  = {f_data.max():.4f}")
print(f"  mean = {f_data.mean():.4f}")
print(f"  std  = {f_data.std():.4f}")

# Sanity: f=0 baseline is r=μ/γ
r2 = lambda y, yh: 1 - np.var(y - yh) / np.var(y)
r_base = mu / gam
print(f"\nBaseline  f = 0  →  r = μ/γ")
print(f"  R²(r)  = {r2(r_data, r_base):.6f}")
print(f"  R²(p₁) = {r2(y1, S*r_base/(1+r_base)):.6f}")
print(f"  R²(p₂) = {r2(y2, S/(1+r_base)):.6f}")
print()

VAR_NAMES = ["lam", "mu", "gam"]

# ─── 5. PySR — polynomial fractions only ─────────────────────────────────────
# No exp / log / sqrt — only rational arithmetic + square (= x², polynomial).
print("=" * 65)
print("PySR — fitting  f = μ/(p₁/p₂) − γ  as a polynomial fraction")
print("  Operators : +  −  *  /  square  (rational functions only)")
print("=" * 65)

model_f = PySRRegressor(
    binary_operators  = ["+", "-", "*", "/"],
    unary_operators   = ["square"],          # x² is polynomial; nothing else
    niterations       = 300,
    populations       = 60,
    population_size   = 50,
    maxsize           = 30,
    parsimony         = 5e-4,               # slightly higher → favour simpler fractions
    turbo             = True,
    temp_equation_file= False,
    random_state      = 42,
    verbosity         = 1,
)

model_f.fit(X, f_data, variable_names=VAR_NAMES)

print("\nPareto front  (f = μ/r − γ):")
print(model_f.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest f expression : {model_f.sympy()}")

# ─── 6. Reconstruct r, p₁, p₂ ───────────────────────────────────────────────
f_pred = model_f.predict(X)
r_pred = mu / (gam + f_pred)           # r = μ/(γ+f)  — core reconstruction
r_pred_safe = np.maximum(r_pred, 1e-9)

p1_pred = S * r_pred_safe / (1.0 + r_pred_safe)
p2_pred = S / (1.0 + r_pred_safe)

print()
print("=" * 65)
print("Constraint verification  (best expression)")
print("=" * 65)
print(f"r > 0            : {np.all(r_pred > 0)}  "
      f"(min r = {r_pred.min():.4f})")
print(f"p₁ ∈ [0,S]       : {np.all((p1_pred >= 0) & (p1_pred <= S))}")
print(f"p₂ ∈ [0,S]       : {np.all((p2_pred >= 0) & (p2_pred <= S))}")
print(f"Sum error  (max) : {np.abs(p1_pred+p2_pred-S).max():.2e}  (exact)")
print()
print(f"R²(f)   = {r2(f_data, f_pred):.6f}")
print(f"R²(r)   = {r2(r_data, r_pred):.6f}")
print(f"R²(p₁)  = {r2(y1, p1_pred):.6f}")
print(f"R²(p₂)  = {r2(y2, p2_pred):.6f}")

# ─── 7. Full Pareto front with R² ────────────────────────────────────────────
print()
print("=" * 65)
print("Full Pareto front  —  R² at each complexity")
print("=" * 65)
print(f"  {'c':>3}  {'loss':>10}  {'R²(f)':>9}  {'R²(r)':>9}  "
      f"{'R²(p₁)':>9}  {'R²(p₂)':>9}  expression")
for iloc_idx, row in enumerate(model_f.equations_.itertuples()):
    f_c  = model_f.predict(X, iloc_idx)
    r_c  = mu / (gam + f_c)
    r_cs = np.maximum(r_c, 1e-9)
    p1_c = S * r_cs / (1.0 + r_cs)
    p2_c = S / (1.0 + r_cs)
    eq_s = str(row.equation)[:55]
    print(f"  {int(row.complexity):>3}  {row.loss:>10.4e}  "
          f"{r2(f_data, f_c):>9.6f}  "
          f"{r2(r_data, r_c):>9.6f}  "
          f"{r2(y1, p1_c):>9.6f}  "
          f"{r2(y2, p2_c):>9.6f}  {eq_s}")

# ─── 8. R² breakdown by μ ────────────────────────────────────────────────────
print()
print("R² breakdown by μ  (best expression):")
print(f"  {'μ':>5}  {'n':>5}  {'R²(f)':>9}  {'R²(r)':>9}  "
      f"{'R²(p₁)':>9}  {'R²(p₂)':>9}")
for mu_v in sorted(np.unique(mu)):
    m = mu == mu_v
    r_m  = np.maximum(mu[m] / (gam[m] + f_pred[m]), 1e-9)
    p1_m = S[m] * r_m / (1.0 + r_m)
    p2_m = S[m] / (1.0 + r_m)
    print(f"  {mu_v:5.0f}  {m.sum():5d}  "
          f"{r2(f_data[m], f_pred[m]):9.6f}  "
          f"{r2(r_data[m], r_m):9.6f}  "
          f"{r2(y1[m], p1_m):9.6f}  "
          f"{r2(y2[m], p2_m):9.6f}")

# ─── 9. Human-readable summary ───────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY  —  r = μ/(γ + f),  p₁ = S·r/(1+r),  p₂ = S/(1+r)")
print("=" * 65)
import sympy as sp
try:
    expr = model_f.sympy()
    lam_s, mu_s, gam_s = sp.symbols("lambda mu gamma", positive=True)
    f_sym = sp.sympify(str(expr).replace("lam","lambda").replace("gam","gamma"))
    r_sym = mu_s / (gam_s + f_sym)
    p1_sym = sp.simplify(r_sym / (1 + r_sym))
    p2_sym = sp.simplify(1 / (1 + r_sym))
    print(f"\nf        = {sp.simplify(f_sym)}")
    print(f"r = p₁/p₂ = {sp.simplify(r_sym)}")
    print(f"p₁/S     = {p1_sym}")
    print(f"p₂/S     = {p2_sym}")
except Exception as e:
    print(f"(sympy simplification skipped: {e})")
    print(f"f = {model_f.sympy()}")
    print(f"r = mu / (gam + f)")

print()
print("─── Selected Pareto entries ───")
print(f"{'c':>3}  {'f expression':50}  {'R²(p₁)':>9}  {'R²(p₂)':>9}")
for iloc_idx, row in enumerate(model_f.equations_.itertuples()):
    f_c  = model_f.predict(X, iloc_idx)
    r_c  = np.maximum(mu / (gam + f_c), 1e-9)
    p1_c = S * r_c / (1 + r_c)
    p2_c = S / (1 + r_c)
    rp1  = r2(y1, p1_c)
    rp2  = r2(y2, p2_c)
    if rp1 > 0.98:   # only show useful entries
        print(f"{int(row.complexity):>3}  {str(row.equation):<50}  "
              f"{rp1:>9.6f}  {rp2:>9.6f}")

# ─── 10. Sample rows ─────────────────────────────────────────────────────────
print()
print("Sample (first 5 rows):")
print(f"{'λ':>6} {'μ':>5} {'γ':>5}  {'f_true':>9}  {'f_pred':>9}  "
      f"{'r_true':>8}  {'r_pred':>8}  {'p₁_true':>9}  {'p₁_pred':>9}")
for i in range(5):
    fi  = f_pred[i]
    ri  = mu[i] / (gam[i] + fi)
    p1i = S[i] * ri / (1 + ri)
    print(f"{lam[i]:6.1f} {mu[i]:5.0f} {gam[i]:5.2f}  "
          f"{f_data[i]:9.4f}  {fi:9.4f}  "
          f"{r_data[i]:8.4f}  {ri:8.4f}  "
          f"{y1[i]:9.5f}  {p1i:9.5f}")
