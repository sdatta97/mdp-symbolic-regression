"""
PySR on merged may21 dataset — direct independent fits for p₁ and p₂.

NO reparametrisation. Fit p₁ and p₂ directly with penalty-based enforcement
of the sum constraint p₁ + p₂ = S  where  S = (λ/γ)·exp(−λ/γ).

Two-pass strategy
─────────────────
Pass 1 — fit p₁ directly:
    target  = y₁
    weights = S  (upper bound on p₁)
    loss    = MSE + 1e7·[p₁<0]² + 1e7·[p₁>S]²

Pass 2 — fit p₂ directly, sum-constrained against p₁:
    target  = y₂
    weights = S − p₁_pred  (ideal p₂ for exact sum)
    loss    = MSE + 1e7·(p₂ − weight)²   ← sum-constraint penalty
                  + 1e7·[p₂<0]²           ← non-negativity

File    : merged_may21_all.csv  (5526 rows)
States  : index 1 (p₁) and index 2 (p₂)
Features: lam, mu, gam
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

print(f"Total rows after cleaning : {len(df)}")
print(f"mu  values : {sorted(df['mu'].unique())}")
print(f"lambda     : {df['lambda'].min():.1f} – {df['lambda'].max():.1f}")
print(f"gamma      : {df['gamma'].min():.2f} – {df['gamma'].max():.1f}")
print()

# ─── 2. Parse state probabilities (mean over all runs) ───────────────────────
def parse_run_arrays(cell: str) -> np.ndarray:
    arrays = []
    for m in re.finditer(r"array\(\[([^\]]*)\]\)", cell):
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1))
        arrays.append([float(x) for x in nums])
    return np.array(arrays)

state_probs = np.array([
    parse_run_arrays(c).mean(axis=0)
    for c in df["results_state_probabilities"]
])   # (5526, 37)

lam_all = df["lambda"].values
mu_all  = df["mu"].values
gam_all = df["gamma"].values
y1_all  = state_probs[:, 1]   # second state  (p₁)
y2_all  = state_probs[:, 2]   # third state   (p₂)

# ─── 3. Analytical sum S ─────────────────────────────────────────────────────
S_all = (lam_all / gam_all) * np.exp(-lam_all / gam_all)

print("=== Sum constraint check (data) ===")
data_sum = y1_all + y2_all
print(f"  max |p1+p2 − S| : {np.abs(data_sum - S_all).max():.3e}")
print(f"  rel. RMSE       : {np.sqrt(np.mean((data_sum-S_all)**2))/S_all[S_all>1e-3].mean():.5f}")
print()

# ─── 4. Filter to S > 0.001 ──────────────────────────────────────────────────
SMIN = 1e-3
mask = S_all > SMIN

X   = np.column_stack([lam_all[mask], mu_all[mask], gam_all[mask]])
S   = S_all[mask]
y1  = y1_all[mask]
y2  = y2_all[mask]

print(f"Training rows (S > {SMIN:.0e}) : {mask.sum()} / {len(df)}")
print(f"  mu breakdown:")
for mu_v in sorted(np.unique(X[:,1])):
    n = (X[:,1] == mu_v).sum()
    print(f"    mu={mu_v:.0f} : {n} rows")
print(f"y₁ range : [{y1.min():.5f}, {y1.max():.5f}]")
print(f"y₂ range : [{y2.min():.5f}, {y2.max():.5f}]")
print(f"S  range : [{S.min():.5f},  {S.max():.5f}]  (max 1/e ≈ 0.3679)")
print()

VAR_NAMES = ["lam", "mu", "gam"]

COMMON = dict(
    binary_operators   = ["+", "-", "*", "/"],
    unary_operators    = ["exp", "log", "sqrt", "square"],
    niterations        = 100,
    populations        = 30,
    population_size    = 50,
    maxsize            = 30,
    parsimony          = 1e-4,
    turbo              = True,
    temp_equation_file = False,
    random_state       = 42,
    verbosity          = 1,
)

r2 = lambda y, yh: 1 - np.var(y - yh) / np.var(y)

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1 — fit p₁ directly   (bounds: 0 ≤ p₁ ≤ S)
# weights = S  (the upper bound, passed into the loss)
# ═══════════════════════════════════════════════════════════════════════════════
LOSS_P1 = """
function loss(prediction, target, weight)
    # weight = S = (lam/gam)*exp(-lam/gam)
    mse = (prediction - target)^2
    lo  = max(zero(prediction), -prediction)^2
    hi  = max(zero(prediction), prediction - weight)^2
    return mse + 1.0e7 * (lo + hi)
end
"""

print("=" * 60)
print("Pass 1  →  fitting  p₁  directly  (target = y₁, bounds [0, S])")
print("=" * 60)

model_p1 = PySRRegressor(elementwise_loss=LOSS_P1, **COMMON)
model_p1.fit(X, y1, weights=S, variable_names=VAR_NAMES)

print("\nPareto front (p₁):")
print(model_p1.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest p₁ expression : {model_p1.sympy()}")

p1_pred = model_p1.predict(X)

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2 — fit p₂ directly, sum-constrained against p₁_pred
# weights = S − p₁_pred  (ideal p₂ for exact sum)
# loss enforces:  p₂ ≈ S − p₁_pred  (sum constraint, heavy penalty)
#                 p₂ ≈ y₂            (MSE against data)
#                 p₂ ≥ 0             (non-negativity)
# ═══════════════════════════════════════════════════════════════════════════════
LOSS_P2 = """
function loss(prediction, target, weight)
    # weight = S - p1_pred  (sum-consistent ideal for p₂)
    mse     = (prediction - target)^2
    sum_pen = (prediction - weight)^2        # enforce p₂ = S − p₁_pred
    lo      = max(zero(prediction), -prediction)^2
    return mse + 1.0e7 * sum_pen + 1.0e7 * lo
end
"""

p2_ideal = S - p1_pred   # per-sample ideal p₂ for exact sum

print()
print("=" * 60)
print("Pass 2  →  fitting  p₂  directly  (sum-constrained vs p₁ preds)")
print("=" * 60)

model_p2 = PySRRegressor(elementwise_loss=LOSS_P2, **COMMON)
model_p2.fit(X, y2, weights=p2_ideal, variable_names=VAR_NAMES)

print("\nPareto front (p₂):")
print(model_p2.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest p₂ expression : {model_p2.sympy()}")

p2_pred = model_p2.predict(X)

# ═══════════════════════════════════════════════════════════════════════════════
# Constraint verification
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Constraint verification")
print("=" * 60)
print(f"p₁ ∈ [0, S]  : {np.all((p1_pred >= 0) & (p1_pred <= S))}")
print(f"p₂ ∈ [0, S]  : {np.all((p2_pred >= 0) & (p2_pred <= S))}")
print(f"p₁ ∈ [0, 1]  : {np.all((p1_pred >= 0) & (p1_pred <= 1))}")
print(f"p₂ ∈ [0, 1]  : {np.all((p2_pred >= 0) & (p2_pred <= 1))}")
print()
sum_err = np.abs(p1_pred + p2_pred - S)
print(f"Sum |p₁+p₂ − S|  max  : {sum_err.max():.4e}")
print(f"Sum |p₁+p₂ − S|  mean : {sum_err.mean():.4e}")
print()
print(f"R²  p₁  : {r2(y1, p1_pred):.6f}")
print(f"R²  p₂  : {r2(y2, p2_pred):.6f}")
print()

# ─── R² breakdown by mu ──────────────────────────────────────────────────────
print("R² breakdown by mu:")
print(f"  {'mu':>5}  {'n':>5}  {'R²(p₁)':>8}  {'R²(p₂)':>8}")
for mu_v in sorted(np.unique(X[:,1])):
    m = X[:,1] == mu_v
    print(f"  {mu_v:5.0f}  {m.sum():5d}  "
          f"{r2(y1[m], p1_pred[m]):8.5f}  "
          f"{r2(y2[m], p2_pred[m]):8.5f}")
print()

# ─── Full Pareto fronts ───────────────────────────────────────────────────────
print("=" * 60)
print("Full Pareto front — p₁")
print("=" * 60)
for _, row in model_p1.equations_.iterrows():
    pred  = model_p1.predict(X, int(row["complexity"]))
    r2_p1 = r2(y1, pred)
    r2_p2 = r2(y2, S - pred)   # implied p₂ = S - p₁
    print(f"  complexity={int(row['complexity']):3d}  loss={row['loss']:.4e}  "
          f"R²(p₁)={r2_p1:.5f}  R²(p₂_implied)={r2_p2:.5f}  eq: {row['equation']}")

print()
print("=" * 60)
print("Full Pareto front — p₂")
print("=" * 60)
for _, row in model_p2.equations_.iterrows():
    pred  = model_p2.predict(X, int(row["complexity"]))
    r2_v  = r2(y2, pred)
    print(f"  complexity={int(row['complexity']):3d}  loss={row['loss']:.4e}  "
          f"R²(p₂)={r2_v:.5f}  eq: {row['equation']}")

# ─── Sample rows ─────────────────────────────────────────────────────────────
print()
print("Sample check (first 5 rows):")
print(f"{'λ':>6} {'μ':>5} {'γ':>5}  {'S':>7}  "
      f"{'p₁_true':>9} {'p₁_pred':>9}  "
      f"{'p₂_true':>9} {'p₂_pred':>9}  {'Σpred':>8}  {'S':>8}")
for i in range(5):
    print(f"{X[i,0]:6.1f} {X[i,1]:5.0f} {X[i,2]:5.2f}  {S[i]:7.4f}  "
          f"{y1[i]:9.5f} {p1_pred[i]:9.5f}  "
          f"{y2[i]:9.5f} {p2_pred[i]:9.5f}  "
          f"{p1_pred[i]+p2_pred[i]:8.6f}  {S[i]:8.6f}")
