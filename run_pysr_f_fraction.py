"""
PySR — fit  r = p₁/p₂  in the form  r = (f·μ) / ((1−f)·μ + γ)
where f is a polynomial fraction (rational function), 0 < f < 1.

Physical interpretation
───────────────────────
  p₁ = S · f·μ / (μ+γ)
  p₂ = S · ((1−f)·μ + γ) / (μ+γ)

  Sum: p₁ + p₂ = S · (f·μ + (1−f)·μ + γ) / (μ+γ) = S  ← exact

  f = 1  →  r = μ/γ  (leading-order baseline)
  f < 1  →  r < μ/γ  (load correction reduces ratio)

Deriving the PySR target
────────────────────────
  From r = (f·μ)/((1−f)·μ+γ)  →  f = r·(μ+γ) / (μ·(1+r))
  Using r = y₁/y₂ from data:
      f_data = y₁·(μ+γ) / (μ·(y₁+y₂))

Operators  : +  −  *  /  square    (polynomial fractions only)
Loss       : MSE + 1e7·[f<0]²  + 1e7·[f>1]²
Dataset    : merged_all_infinite_buffer.csv  (6994 rows)
             μ ∈ {50, 100, 125, 150, 175, 200}
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

# ─── 4. Compute target  f = r(μ+γ) / (μ(1+r)) ───────────────────────────────
r_data    = y1 / y2
f_data    = r_data * (mu + gam) / (mu * (1.0 + r_data))
# equivalently:  f_data = y1*(mu+gam) / (mu*(y1+y2))

print(f"Training rows  (S>1e-3, y₁>1e-3, y₂>1e-3) : {mask.sum()} / {len(df)}")
print(f"μ breakdown:")
for mu_v in sorted(np.unique(mu)):
    print(f"  μ={int(mu_v):3d} : {(mu==mu_v).sum()} rows")

pct_above1 = 100 * np.mean(f_data > 1)
pct_below0 = 100 * np.mean(f_data < 0)
print(f"\nf_data statistics:")
print(f"  min    = {f_data.min():.5f}")
print(f"  max    = {f_data.max():.5f}")
print(f"  mean   = {f_data.mean():.5f}")
print(f"  median = {np.median(f_data):.5f}")
print(f"  f > 1  : {pct_above1:.2f}% of rows  (r > μ/γ)")
print(f"  f < 0  : {pct_below0:.2f}% of rows")

r2 = lambda y, yh: 1 - np.var(y - yh) / np.var(y)

# Baseline: f = 1 → r = μ/γ
r_base = mu / gam
p1_base = S * r_base / (1 + r_base)
p2_base = S / (1 + r_base)
print(f"\nBaseline  f = 1  →  r = μ/γ:")
print(f"  R²(r)  = {r2(r_data, r_base):.6f}")
print(f"  R²(p₁) = {r2(y1, p1_base):.6f}")
print(f"  R²(p₂) = {r2(y2, p2_base):.6f}")
print()

VAR_NAMES = ["lam", "mu", "gam"]

# ─── 5. PySR ─────────────────────────────────────────────────────────────────
LOSS_F = """
function loss(prediction, target)
    mse = (prediction - target)^2
    lo  = max(zero(prediction), -prediction)^2
    hi  = max(zero(prediction), prediction - one(prediction))^2
    return mse + 1.0e7 * (lo + hi)
end
"""

print("=" * 65)
print("PySR — fitting  f  in  r = (f·μ) / ((1−f)·μ + γ)")
print("  Operators: +  −  *  /  square   (polynomial fractions)")
print("  Constraint: 0 < f < 1  (1e7 penalty)")
print("=" * 65)

model_f = PySRRegressor(
    binary_operators   = ["+", "-", "*", "/"],
    unary_operators    = ["square"],
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
f_pred = model_f.predict(X)
f_pred = np.clip(f_pred, 1e-9, 1 - 1e-9)       # safety clip

r_pred = (f_pred * mu) / ((1 - f_pred) * mu + gam)
p1_pred = S * r_pred / (1.0 + r_pred)
p2_pred = S / (1.0 + r_pred)

# ─── 7. Verification ─────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Constraint verification  (best expression)")
print("=" * 65)
print(f"f  ∈ (0,1)  : {np.all((f_pred > 0) & (f_pred < 1))}")
print(f"r  > 0      : {np.all(r_pred > 0)}")
print(f"p₁ ∈ [0,S]  : {np.all((p1_pred >= 0) & (p1_pred <= S))}")
print(f"p₂ ∈ [0,S]  : {np.all((p2_pred >= 0) & (p2_pred <= S))}")
print(f"Sum (max)   : {np.abs(p1_pred+p2_pred-S).max():.2e}  (exact by construction)")
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
    f_c  = np.clip(model_f.predict(X, iloc_idx), 1e-9, 1-1e-9)
    r_c  = (f_c * mu) / ((1 - f_c) * mu + gam)
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
    m = mu == mu_v
    f_m  = np.clip(model_f.predict(X[m]), 1e-9, 1-1e-9)
    r_m  = (f_m * mu[m]) / ((1 - f_m) * mu[m] + gam[m])
    p1_m = S[m] * r_m / (1.0 + r_m)
    p2_m = S[m] / (1.0 + r_m)
    print(f"  {int(mu_v):5d}  {m.sum():5d}  "
          f"{r2(f_data[m], f_m):9.6f}  "
          f"{r2(r_data[m], r_m):9.6f}  "
          f"{r2(y1[m], p1_m):9.6f}  "
          f"{r2(y2[m], p2_m):9.6f}")

# ─── 10. Final summary ───────────────────────────────────────────────────────
print()
print("=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"\n  r = (f·μ) / ((1−f)·μ + γ)")
print(f"  f = {model_f.sympy()}")
print(f"\n  p₁ = S · f·μ / (μ+γ)")
print(f"  p₂ = S · ((1−f)·μ+γ) / (μ+γ)")
print(f"  Sum constraint: exact")
print(f"\n  R²(p₁) = {r2(y1, p1_pred):.6f}")
print(f"  R²(p₂) = {r2(y2, p2_pred):.6f}")

# ─── 11. Sample rows ─────────────────────────────────────────────────────────
print()
print("Sample (first 5 rows):")
print(f"{'λ':>6} {'μ':>5} {'γ':>5}  {'f_true':>8}  {'f_pred':>8}  "
      f"{'r_true':>8}  {'r_pred':>8}  {'p₁_true':>9}  {'p₁_pred':>9}")
for i in range(5):
    fi  = float(np.clip(model_f.predict(X[[i]]), 1e-9, 1-1e-9))
    ri  = (fi * mu[i]) / ((1 - fi) * mu[i] + gam[i])
    p1i = S[i] * ri / (1 + ri)
    print(f"{X[i,0]:6.1f} {X[i,1]:5.0f} {X[i,2]:5.2f}  "
          f"{f_data[i]:8.5f}  {fi:8.5f}  "
          f"{r_data[i]:8.4f}  {ri:8.4f}  "
          f"{y1[i]:9.5f}  {p1i:9.5f}")
