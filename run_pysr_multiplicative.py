"""
PySR — fit  r = p₁/p₂  in the form  r = (1 + f) · μ/γ
where f is a polynomial fraction (rational function).

Structure
─────────
  r = (1+f) · μ/γ
  f = 0  →  r = μ/γ  (baseline, complexity-0 correction)
  f > 0  →  r > μ/γ
  f > -1 required for r > 0

  p₁ = S · r/(1+r)  = S·(1+f)·μ / ((1+f)·μ + γ)
  p₂ = S · 1/(1+r)  = S·γ        / ((1+f)·μ + γ)
  Sum p₁+p₂ = S  ←  exact by construction

Target
──────
  f_data = r·(γ/μ) − 1  =  (y₁/y₂)·(γ/μ) − 1

Operators : +  −  *  square   (pure polynomials ONLY — no division)
Loss      : MSE  +  1e7·[f < −1]²   (only lower bound matters)
Dataset   : merged_all_infinite_buffer.csv  (6994 rows)
"""

import re as _re
import numpy as np
import pandas as pd
from pysr import PySRRegressor

# ─── 1. Load & clean ─────────────────────────────────────────────────────────
df = pd.read_csv("/Users/sdatta/Downloads/Claude/merged_all_infinite_buffer.csv")
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
SMIN = 1e-3;  YMIN = 1e-3
mask = (S_all > SMIN) & (y1_all > YMIN) & (y2_all > YMIN)

X   = np.column_stack([lam_all[mask], mu_all[mask], gam_all[mask]])
S   = S_all[mask]
y1  = y1_all[mask];  y2 = y2_all[mask]
lam = X[:,0];  mu = X[:,1];  gam = X[:,2]

# ─── 4. Compute target  f = r·(γ/μ) − 1 ─────────────────────────────────────
r_data = y1 / y2
f_data = r_data * (gam / mu) - 1.0

print(f"Training rows : {mask.sum()} / {len(df)}")
print(f"μ breakdown:")
for mu_v in sorted(np.unique(mu)):
    print(f"  μ={int(mu_v):3d} : {(mu==mu_v).sum()} rows")

print(f"\nf_data  =  r·γ/μ − 1  statistics:")
print(f"  min    = {f_data.min():.5f}")
print(f"  max    = {f_data.max():.5f}")
print(f"  mean   = {f_data.mean():.5f}")
print(f"  median = {np.median(f_data):.5f}")
print(f"  f > 0  : {100*np.mean(f_data > 0):.1f}%  (r > μ/γ)")
print(f"  f < -1 : {100*np.mean(f_data < -1):.1f}%  (should be 0)")

r2 = lambda y, yh: 1 - np.var(y - yh) / np.var(y)

# Baseline f=0 → r=μ/γ
r_base = mu / gam
p1_base = S * r_base / (1 + r_base)
p2_base = S / (1 + r_base)
print(f"\nBaseline  f = 0  →  r = μ/γ:")
print(f"  R²(r)  = {r2(r_data, r_base):.6f}")
print(f"  R²(p₁) = {r2(y1, p1_base):.6f}")
print(f"  R²(p₂) = {r2(y2, p2_base):.6f}")
print()

VAR_NAMES = ["lam", "mu", "gam"]

# ─── 5. PySR ─────────────────────────────────────────────────────────────────
# Only constraint: f > -1  (so that r = (1+f)·μ/γ > 0)
LOSS_F = """
function loss(prediction, target)
    mse = (prediction - target)^2
    lo  = max(zero(prediction), -(prediction + one(prediction)))^2
    return mse + 1.0e7 * lo
end
"""

print("=" * 65)
print("PySR — fitting  f  in  r = (1+f)·μ/γ")
print("  Operators: +  −  *  square   (pure polynomials ONLY — no division)")
print("  Constraint: f > −1  (1e7 penalty)")
print("=" * 65)

model_f = PySRRegressor(
    binary_operators   = ["+", "-", "*"],       # NO division — pure polynomials
    unary_operators    = ["square"],             # x² is polynomial
    elementwise_loss   = LOSS_F,
    niterations        = 300,
    populations        = 60,
    population_size    = 50,
    maxsize            = 30,
    parsimony          = 5e-4,
    turbo              = True,
    temp_equation_file = False,
    random_state       = 42,
    verbosity          = 1,
)

model_f.fit(X, f_data, variable_names=VAR_NAMES)

print("\nPareto front  (f):")
print(model_f.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest f : {model_f.sympy()}")

# ─── 6. Reconstruct r, p₁, p₂ ───────────────────────────────────────────────
f_pred  = model_f.predict(X)
r_pred  = (1.0 + f_pred) * mu / gam
r_safe  = np.maximum(r_pred, 1e-9)
p1_pred = S * r_safe / (1.0 + r_safe)
p2_pred = S / (1.0 + r_safe)

# ─── 7. Verification ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Constraint verification  (best expression)")
print("=" * 65)
print(f"f  > −1     : {np.all(f_pred > -1)}  (min f = {f_pred.min():.4f})")
print(f"r  > 0      : {np.all(r_pred > 0)}")
print(f"p₁ ∈ [0,1]  : {np.all((p1_pred >= 0) & (p1_pred <= 1))}")
print(f"p₂ ∈ [0,1]  : {np.all((p2_pred >= 0) & (p2_pred <= 1))}")
print(f"Sum (max)   : {np.abs(p1_pred+p2_pred-S).max():.2e}  (exact)")
print()
print(f"R²(f)   = {r2(f_data, f_pred):.6f}")
print(f"R²(r)   = {r2(r_data, r_pred):.6f}")
print(f"R²(p₁)  = {r2(y1, p1_pred):.6f}")
print(f"R²(p₂)  = {r2(y2, p2_pred):.6f}")

# ─── 8. Full Pareto R² table ─────────────────────────────────────────────────
print()
print("=" * 65)
print("Full Pareto front  —  R² at each complexity")
print("=" * 65)
print(f"  {'c':>3}  {'loss':>10}  {'R²(f)':>9}  {'R²(r)':>9}  "
      f"{'R²(p₁)':>9}  {'R²(p₂)':>9}  expression")
for iloc_idx, row in enumerate(model_f.equations_.itertuples()):
    f_c  = model_f.predict(X, iloc_idx)
    r_c  = np.maximum((1.0 + f_c) * mu / gam, 1e-9)
    p1_c = S * r_c / (1.0 + r_c)
    p2_c = S / (1.0 + r_c)
    eq_s = str(row.equation)[:55]
    print(f"  {int(row.complexity):>3}  {row.loss:>10.4e}  "
          f"{r2(f_data, f_c):>9.6f}  "
          f"{r2(r_data, r_c):>9.6f}  "
          f"{r2(y1, p1_c):>9.6f}  "
          f"{r2(y2, p2_c):>9.6f}  {eq_s}")

# ─── 9. R² breakdown by μ ────────────────────────────────────────────────────
print()
print("R² breakdown by μ  (best expression):")
print(f"  {'μ':>5}  {'n':>5}  {'R²(f)':>9}  {'R²(r)':>9}  "
      f"{'R²(p₁)':>9}  {'R²(p₂)':>9}")
for mu_v in sorted(np.unique(mu)):
    m   = mu == mu_v
    f_m = model_f.predict(X[m])
    r_m = np.maximum((1.0 + f_m) * mu[m] / gam[m], 1e-9)
    p1_m = S[m] * r_m / (1.0 + r_m)
    p2_m = S[m] / (1.0 + r_m)
    print(f"  {int(mu_v):5d}  {m.sum():5d}  "
          f"{r2(f_data[m], f_m):9.6f}  "
          f"{r2(r_data[m], r_m):9.6f}  "
          f"{r2(y1[m], p1_m):9.6f}  "
          f"{r2(y2[m], p2_m):9.6f}")

# ─── 10. Summary ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"\n  r = (1 + f) · μ/γ")
print(f"  f = {model_f.sympy()}")
print(f"\n  p₁ = S · (1+f)·μ / ((1+f)·μ + γ)")
print(f"  p₂ = S · γ        / ((1+f)·μ + γ)")
print(f"  Sum constraint: exact")
print(f"\n  R²(p₁) = {r2(y1, p1_pred):.6f}")
print(f"  R²(p₂) = {r2(y2, p2_pred):.6f}")

print()
print("Sample (first 5 rows):")
print(f"{'λ':>6} {'μ':>5} {'γ':>5}  {'f_true':>8}  {'f_pred':>8}  "
      f"{'r_true':>8}  {'r_pred':>8}  {'p₁_true':>9}  {'p₁_pred':>9}")
for i in range(5):
    fi  = float(model_f.predict(X[[i]]))
    ri  = max((1.0 + fi) * mu[i] / gam[i], 1e-9)
    p1i = S[i] * ri / (1 + ri)
    print(f"{X[i,0]:6.1f} {X[i,1]:5.0f} {X[i,2]:5.2f}  "
          f"{f_data[i]:8.5f}  {fi:8.5f}  "
          f"{r_data[i]:8.4f}  {ri:8.4f}  "
          f"{y1[i]:9.5f}  {p1i:9.5f}")
