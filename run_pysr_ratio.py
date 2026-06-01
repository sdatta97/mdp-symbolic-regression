"""
PySR — fit r = p₁/p₂  (the probability ratio) directly.

Motivation
──────────
  r = p₁/p₂  is unconstrained except r > 0.
  Once r is known, recover probabilities exactly as:

      p₁ = S · r / (1 + r)
      p₂ = S · 1 / (1 + r)

  where  S = (λ/γ)·e^(−λ/γ)

  Sum constraint:  p₁ + p₂ = S  →  exact by arithmetic identity.
  Probability bounds:  p₁, p₂ ∈ (0, S) ⊂ (0, 1/e) ⊂ (0,1)  whenever r > 0.

  From prior reparametrisation work the ratio should be ≈ μ/γ  (complexity 3).
  Fitting r directly gives PySR the cleanest possible signal.

Two variants tried in parallel
───────────────────────────────
  Variant A — fit r = p₁/p₂  directly   (loss: MSE + 1e7·[r<0]²)
  Variant B — fit log(r) = log(p₁/p₂)  (unconstrained; no penalty needed;
                                         additive corrections easier to find)

Dataset : merged_may21_all.csv  (5526 rows, μ ∈ {50,100,125,150,175,200})
Filter  : S > 0.001  AND  y₂ > 0.001  (avoid noisy ratios where p₂ ≈ 0)
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
print(f"mu values  : {sorted(df['mu'].unique())}")
print()

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

S_all = (lam_all / gam_all) * np.exp(-lam_all / gam_all)

# ─── 3. Filter: S > 0.001  AND  y₂ > 0.001 ──────────────────────────────────
SMIN  = 1e-3
Y2MIN = 1e-3    # avoid extremely noisy ratios
mask  = (S_all > SMIN) & (y2_all > Y2MIN)

X   = np.column_stack([lam_all[mask], mu_all[mask], gam_all[mask]])
S   = S_all[mask]
y1  = y1_all[mask]
y2  = y2_all[mask]
lam = X[:, 0]
mu  = X[:, 1]
gam = X[:, 2]

# Ratio and log-ratio targets
r_data    = y1 / y2
logr_data = np.log(r_data)

print(f"Training rows  (S>1e-3 & y₂>1e-3) : {mask.sum()} / {len(df)}")
print(f"  mu breakdown:")
for mu_v in sorted(np.unique(mu)):
    n = (mu == mu_v).sum()
    print(f"    mu={mu_v:.0f} : {n} rows")
print()
print(f"r  = p₁/p₂  : min={r_data.min():.4f}  max={r_data.max():.2f}  "
      f"mean={r_data.mean():.4f}  median={np.median(r_data):.4f}")
print(f"log(r)       : min={logr_data.min():.4f}  max={logr_data.max():.4f}  "
      f"mean={logr_data.mean():.4f}")
print()
print(f"Baseline check — r ≈ μ/γ:")
r_base = mu / gam
print(f"  R² for r ≈ μ/γ          : "
      f"{1 - np.var(r_data - r_base)/np.var(r_data):.6f}")
print(f"  R² for log(r) ≈ log(μ/γ): "
      f"{1 - np.var(logr_data - np.log(r_base))/np.var(logr_data):.6f}")
print()

VAR_NAMES = ["lam", "mu", "gam"]
r2 = lambda y, yh: 1 - np.var(y - yh) / np.var(y)

COMMON = dict(
    binary_operators   = ["+", "-", "*", "/"],
    unary_operators    = ["exp", "log", "sqrt", "square"],
    niterations        = 200,
    populations        = 50,
    population_size    = 50,
    maxsize            = 30,
    parsimony          = 1e-4,
    turbo              = True,
    temp_equation_file = False,
    random_state       = 42,
    verbosity          = 1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT A — fit  r = p₁/p₂  directly
#   Loss: MSE on r + heavy penalty for r ≤ 0
# ═══════════════════════════════════════════════════════════════════════════════
LOSS_R = """
function loss(prediction, target)
    mse = (prediction - target)^2
    lo  = max(zero(prediction), -prediction + 1.0e-6)^2   # enforce r > 0
    return mse + 1.0e7 * lo
end
"""

print("=" * 65)
print("Variant A — fitting  r = p₁/p₂  directly")
print("=" * 65)

model_r = PySRRegressor(elementwise_loss=LOSS_R, **COMMON)
model_r.fit(X, r_data, variable_names=VAR_NAMES)

print("\nPareto front (r = p₁/p₂):")
print(model_r.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest r expression : {model_r.sympy()}")

r_pred = model_r.predict(X)
r_pred = np.maximum(r_pred, 1e-9)      # safety clip

p1_from_r = S * r_pred / (1.0 + r_pred)
p2_from_r = S * 1.0    / (1.0 + r_pred)

print()
print("─── Variant A results ───")
print(f"R²(r)           : {r2(r_data, r_pred):.6f}")
print(f"R²(p₁)          : {r2(y1, p1_from_r):.6f}")
print(f"R²(p₂)          : {r2(y2, p2_from_r):.6f}")
sum_err_A = np.abs(p1_from_r + p2_from_r - S)
print(f"Sum error (max) : {sum_err_A.max():.2e}  (exact by construction)")
print(f"r > 0           : {np.all(r_pred > 0)}")
print(f"p₁ ∈ [0,S]      : {np.all((p1_from_r >= 0) & (p1_from_r <= S))}")
print(f"p₂ ∈ [0,S]      : {np.all((p2_from_r >= 0) & (p2_from_r <= S))}")

print("\nPareto R² — Variant A (r = p₁/p₂):")
print(f"  {'c':>3}  {'loss':>10}  {'R²(r)':>9}  {'R²(p₁)':>9}  {'R²(p₂)':>9}  equation")
for iloc_idx, row in enumerate(model_r.equations_.itertuples()):
    pred_c = np.maximum(model_r.predict(X, iloc_idx), 1e-9)
    p1_c   = S * pred_c / (1.0 + pred_c)
    p2_c   = S / (1.0 + pred_c)
    eq_s   = str(row.equation)[:55]
    print(f"  {int(row.complexity):>3}  {row.loss:>10.4e}  "
          f"{r2(r_data, pred_c):>9.6f}  "
          f"{r2(y1, p1_c):>9.6f}  "
          f"{r2(y2, p2_c):>9.6f}  {eq_s}")

print()
print("R² breakdown by μ — Variant A:")
print(f"  {'mu':>5}  {'n':>5}  {'R²(r)':>9}  {'R²(p₁)':>9}  {'R²(p₂)':>9}")
for mu_v in sorted(np.unique(mu)):
    m = mu == mu_v
    p1m = p1_from_r[m]; p2m = p2_from_r[m]
    print(f"  {mu_v:5.0f}  {m.sum():5d}  "
          f"{r2(r_data[m], r_pred[m]):9.6f}  "
          f"{r2(y1[m], p1m):9.6f}  "
          f"{r2(y2[m], p2m):9.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT B — fit  log(r) = log(p₁/p₂)
#   Unconstrained (exp(anything) > 0), additive corrections in log space.
#   Recover: r = exp(log_r_pred),  p₁ = S·r/(1+r),  p₂ = S/(1+r)
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("Variant B — fitting  log(r) = log(p₁/p₂)  [unconstrained]")
print("=" * 65)

model_logr = PySRRegressor(**COMMON)    # plain MSE, no penalty needed
model_logr.fit(X, logr_data, variable_names=VAR_NAMES)

print("\nPareto front (log(r) = log(p₁/p₂)):")
print(model_logr.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest log(r) expression : {model_logr.sympy()}")

logr_pred = model_logr.predict(X)
r_from_logr  = np.exp(logr_pred)
p1_from_logr = S * r_from_logr / (1.0 + r_from_logr)
p2_from_logr = S / (1.0 + r_from_logr)

print()
print("─── Variant B results ───")
print(f"R²(log r)       : {r2(logr_data, logr_pred):.6f}")
print(f"R²(p₁)          : {r2(y1, p1_from_logr):.6f}")
print(f"R²(p₂)          : {r2(y2, p2_from_logr):.6f}")
sum_err_B = np.abs(p1_from_logr + p2_from_logr - S)
print(f"Sum error (max) : {sum_err_B.max():.2e}  (exact by construction)")

print("\nPareto R² — Variant B (log r = log(p₁/p₂)):")
print(f"  {'c':>3}  {'loss':>10}  {'R²(lr)':>9}  {'R²(p₁)':>9}  {'R²(p₂)':>9}  equation")
for iloc_idx, row in enumerate(model_logr.equations_.itertuples()):
    logr_c  = model_logr.predict(X, iloc_idx)
    r_c     = np.exp(logr_c)
    p1_c    = S * r_c / (1.0 + r_c)
    p2_c    = S / (1.0 + r_c)
    eq_s    = str(row.equation)[:55]
    print(f"  {int(row.complexity):>3}  {row.loss:>10.4e}  "
          f"{r2(logr_data, logr_c):>9.6f}  "
          f"{r2(y1, p1_c):>9.6f}  "
          f"{r2(y2, p2_c):>9.6f}  {eq_s}")

print()
print("R² breakdown by μ — Variant B:")
print(f"  {'mu':>5}  {'n':>5}  {'R²(lr)':>9}  {'R²(p₁)':>9}  {'R²(p₂)':>9}")
for mu_v in sorted(np.unique(mu)):
    m = mu == mu_v
    p1m = p1_from_logr[m]; p2m = p2_from_logr[m]
    print(f"  {mu_v:5.0f}  {m.sum():5d}  "
          f"{r2(logr_data[m], logr_pred[m]):9.6f}  "
          f"{r2(y1[m], p1m):9.6f}  "
          f"{r2(y2[m], p2m):9.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print()
print(f"  Known baseline  r = μ/γ  →  p₁ = Sμ/(μ+γ),  p₂ = Sγ/(μ+γ)")
r_base   = mu / gam
p1_base  = S * r_base / (1.0 + r_base)
p2_base  = S / (1.0 + r_base)
print(f"    R²(p₁) = {r2(y1, p1_base):.6f},  R²(p₂) = {r2(y2, p2_base):.6f}")
print()
print(f"  Variant A best  r = {model_r.sympy()}")
print(f"    R²(p₁) = {r2(y1, p1_from_r):.6f},  R²(p₂) = {r2(y2, p2_from_r):.6f}")
print()
print(f"  Variant B best  log(r) = {model_logr.sympy()}")
print(f"    R²(p₁) = {r2(y1, p1_from_logr):.6f},  R²(p₂) = {r2(y2, p2_from_logr):.6f}")
print()
print("  Sum constraint: EXACT for all variants (arithmetic identity)")

# ─── Sample rows ─────────────────────────────────────────────────────────────
print()
print("Sample check (first 5 rows):")
print(f"{'λ':>6} {'μ':>5} {'γ':>5}  {'r_true':>8}  {'r_A':>8}  {'r_B':>8}  "
      f"{'p₁_true':>9}  {'p₁_A':>9}  {'p₁_B':>9}")
for i in range(5):
    r_true = r_data[i]
    rA = r_pred[i]
    rB = r_from_logr[i]
    print(f"{X[i,0]:6.1f} {X[i,1]:5.0f} {X[i,2]:5.2f}  "
          f"{r_true:8.4f}  {rA:8.4f}  {rB:8.4f}  "
          f"{y1[i]:9.5f}  {p1_from_r[i]:9.5f}  {p1_from_logr[i]:9.5f}")
