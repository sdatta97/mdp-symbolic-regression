"""
PySR on merged may21 — target R² ≥ 0.999 for p₁ and p₂.

Strategy
────────
• p₁ : direct fit with 300 iterations, 50 populations (3× budget)
• p₂ : two approaches reported side-by-side
    (a) implied  — p₂ = S − p₁_pred  (exact sum, propagates p₁ accuracy)
    (b) direct   — independent fit with bounds [0,S], no sum penalty
        (lets PySR optimise p₂ on its own merit)

Bugfix vs previous run
───────────────────────
  model.predict(X, index) uses iloc position, NOT complexity value.
  Pareto-front loop now uses enumerate() to pass the correct iloc index.
"""

import re
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
    for m in re.finditer(r"array\(\[([^\]]*)\]\)", cell):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))
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

# ─── 3. Filter ───────────────────────────────────────────────────────────────
SMIN = 1e-3
mask = S_all > SMIN

X   = np.column_stack([lam_all[mask], mu_all[mask], gam_all[mask]])
S   = S_all[mask]
y1  = y1_all[mask]
y2  = y2_all[mask]

print(f"Training rows : {mask.sum()} / {len(df)}")
for mu_v in sorted(np.unique(X[:,1])):
    n = (X[:,1] == mu_v).sum()
    print(f"  mu={mu_v:.0f} : {n} rows")
print()

VAR_NAMES = ["lam", "mu", "gam"]
r2 = lambda y, yh: 1 - np.var(y - yh) / np.var(y)

COMMON = dict(
    binary_operators   = ["+", "-", "*", "/"],
    unary_operators    = ["exp", "log", "sqrt", "square"],
    niterations        = 300,        # 3× budget vs previous run
    populations        = 50,         # more independent populations
    population_size    = 50,
    maxsize            = 35,
    parsimony          = 1e-4,
    turbo              = True,
    temp_equation_file = False,
    random_state       = 42,
    verbosity          = 1,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1 — p₁  direct fit   (bounds: 0 ≤ p₁ ≤ S)
# ═══════════════════════════════════════════════════════════════════════════════
LOSS_P1 = """
function loss(prediction, target, weight)
    mse = (prediction - target)^2
    lo  = max(zero(prediction), -prediction)^2
    hi  = max(zero(prediction), prediction - weight)^2
    return mse + 1.0e7 * (lo + hi)
end
"""

print("=" * 65)
print("Pass 1 — p₁  direct fit  (300 iterations, 50 populations)")
print("=" * 65)

model_p1 = PySRRegressor(elementwise_loss=LOSS_P1, **COMMON)
model_p1.fit(X, y1, weights=S, variable_names=VAR_NAMES)

print("\nPareto front (p₁):")
print(model_p1.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest p₁  : {model_p1.sympy()}")

p1_pred = model_p1.predict(X)

# ─── Full Pareto: per-complexity R² (fixed iloc indexing) ────────────────────
print()
print("=" * 65)
print("Pareto R² — p₁  (each row = expression at that complexity)")
print("=" * 65)
print(f"  {'c':>3}  {'loss':>10}  {'R²(p₁)':>9}  {'R²(p₂_impl)':>12}  equation")
for iloc_idx, row in enumerate(model_p1.equations_.itertuples()):
    pred_c  = model_p1.predict(X, iloc_idx)            # ← correct: iloc position
    r2_p1   = r2(y1, pred_c)
    r2_p2i  = r2(y2, S - pred_c)
    eq_short = str(row.equation)[:70]
    print(f"  {int(row.complexity):>3}  {row.loss:>10.4e}  {r2_p1:>9.6f}  "
          f"{r2_p2i:>12.6f}  {eq_short}")

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2a — p₂  implied  (exact sum)
# ═══════════════════════════════════════════════════════════════════════════════
p2_implied = S - p1_pred

print()
print("=" * 65)
print("Pass 2a — p₂ implied = S − p₁_pred  (exact sum constraint)")
print("=" * 65)
print(f"R²(p₁)         : {r2(y1, p1_pred):.6f}")
print(f"R²(p₂_implied) : {r2(y2, p2_implied):.6f}")
print(f"p₁ ∈ [0,S]     : {np.all((p1_pred >= 0) & (p1_pred <= S))}")
print(f"p₂_impl ∈ [0,S]: {np.all((p2_implied >= 0) & (p2_implied <= S))}")
print(f"Sum error (max) : {np.abs(p1_pred + p2_implied - S).max():.2e}  (exact by construction)")

print("\nR² breakdown (implied p₂ = S − p₁):")
print(f"  {'mu':>5}  {'n':>5}  {'R²(p₁)':>9}  {'R²(p₂_impl)':>12}")
for mu_v in sorted(np.unique(X[:,1])):
    m = X[:,1] == mu_v
    print(f"  {mu_v:5.0f}  {m.sum():5d}  {r2(y1[m], p1_pred[m]):9.6f}  "
          f"{r2(y2[m], p2_implied[m]):12.6f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2b — p₂  independent direct fit  (bounds [0,S], no sum penalty)
# ═══════════════════════════════════════════════════════════════════════════════
LOSS_P2_DIRECT = """
function loss(prediction, target, weight)
    # weight = S  (upper bound only — no sum constraint penalty)
    mse = (prediction - target)^2
    lo  = max(zero(prediction), -prediction)^2
    hi  = max(zero(prediction), prediction - weight)^2
    return mse + 1.0e7 * (lo + hi)
end
"""

print()
print("=" * 65)
print("Pass 2b — p₂  independent direct fit  (bounds [0,S] only)")
print("=" * 65)

model_p2 = PySRRegressor(elementwise_loss=LOSS_P2_DIRECT, **COMMON)
model_p2.fit(X, y2, weights=S, variable_names=VAR_NAMES)

print("\nPareto front (p₂):")
print(model_p2.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest p₂  : {model_p2.sympy()}")

p2_pred = model_p2.predict(X)

print()
print("=" * 65)
print("Pareto R² — p₂  (direct fit)")
print("=" * 65)
print(f"  {'c':>3}  {'loss':>10}  {'R²(p₂)':>9}  equation")
for iloc_idx, row in enumerate(model_p2.equations_.itertuples()):
    pred_c = model_p2.predict(X, iloc_idx)
    r2_p2  = r2(y2, pred_c)
    eq_short = str(row.equation)[:70]
    print(f"  {int(row.complexity):>3}  {row.loss:>10.4e}  {r2_p2:>9.6f}  {eq_short}")

# ─── Final summary ────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print()
print("─── Option A: p₁ direct  +  p₂ = S − p₁  (exact sum) ───")
print(f"  R²(p₁)              : {r2(y1, p1_pred):.6f}")
print(f"  R²(p₂)              : {r2(y2, p2_implied):.6f}")
print(f"  Sum constraint      : exact (machine precision)")
print(f"  p₁ expression       : {model_p1.sympy()}")
print(f"  p₂ expression       : S - p₁")
print()
print("─── Option B: p₁ direct  +  p₂ direct  (independent) ───")
print(f"  R²(p₁)              : {r2(y1, p1_pred):.6f}")
print(f"  R²(p₂)              : {r2(y2, p2_pred):.6f}")
sum_err = np.abs(p1_pred + p2_pred - S)
print(f"  Sum |p₁+p₂−S| max   : {sum_err.max():.4e}")
print(f"  Sum |p₁+p₂−S| mean  : {sum_err.mean():.4e}")
print(f"  p₁ expression       : {model_p1.sympy()}")
print(f"  p₂ expression       : {model_p2.sympy()}")
print()

# ─── Sample rows ─────────────────────────────────────────────────────────────
print("Sample check (first 5 rows):")
print(f"{'λ':>6} {'μ':>5} {'γ':>5}  {'S':>7}  "
      f"{'p₁_true':>9} {'p₁_pred':>9}  "
      f"{'p₂_true':>9} {'p₂_impl':>9}  {'p₂_dir':>9}")
for i in range(5):
    print(f"{X[i,0]:6.1f} {X[i,1]:5.0f} {X[i,2]:5.2f}  {S[i]:7.4f}  "
          f"{y1[i]:9.5f} {p1_pred[i]:9.5f}  "
          f"{y2[i]:9.5f} {p2_implied[i]:9.5f}  {p2_pred[i]:9.5f}")
