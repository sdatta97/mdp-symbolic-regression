"""
PySR on the merged may21 dataset — exact sum-constraint via reparametrisation.

File    : merged_may21_all.csv  (5526 rows, mu ∈ {50,100,125,150,175,200})
States  : index 1 (second state) and index 2 (third state)
Average : mean over all simulation runs per row (5 or 10 depending on source)
Features: lambda, mu, gamma   ← mu now varies, enabling full mu dependence
Constraints:
    p₁ + p₂ = S  where  S = (λ/γ)·exp(−λ/γ)      [sum constraint — exact]
    0 ≤ p₁ ≤ 1  and  0 ≤ p₂ ≤ 1                   [probability bounds — exact]

Reparametrisation (same strategy that worked before):
    g = p₁ / S  ∈ [0,1]
    →  p₁ = g·S,   p₂ = (1−g)·S
    sum constraint: p₁+p₂ = S  (arithmetic identity)
    bounds:         0 ≤ pᵢ ≤ S ≤ 1/e < 1  whenever  g ∈ [0,1]
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
y1_all  = state_probs[:, 1]   # second state
y2_all  = state_probs[:, 2]   # third state

# ─── 3. Analytical sum and ratio ─────────────────────────────────────────────
S_all = (lam_all / gam_all) * np.exp(-lam_all / gam_all)

# Verify sum constraint against data
data_sum = y1_all + y2_all
print("=== Sum constraint check ===")
print(f"  max |p1+p2 − S| : {np.abs(data_sum - S_all).max():.3e}")
print(f"  rel. RMSE       : {np.sqrt(np.mean((data_sum-S_all)**2))/S_all[S_all>1e-3].mean():.5f}")
print()

# ─── 4. Filter to S > 0.001 (noise-safe region) ──────────────────────────────
SMIN = 1e-3
mask = S_all > SMIN

X  = np.column_stack([lam_all[mask], mu_all[mask], gam_all[mask]])
S  = S_all[mask]
y1 = y1_all[mask]
y2 = y2_all[mask]

# g = p₁/S — clipped to [0,1] to absorb tiny simulation-noise overshoots
g_data = np.clip(y1 / S, 0.0, 1.0)

print(f"Training rows (S > {SMIN:.0e}) : {mask.sum()} / {len(df)}")
print(f"  mu breakdown:")
for mu_v in sorted(np.unique(X[:,1])):
    n = (X[:,1] == mu_v).sum()
    print(f"    mu={mu_v:.0f} : {n} rows")
print(f"g = p₁/S  :  min={g_data.min():.4f}  max={g_data.max():.4f}  mean={g_data.mean():.4f}")
print(f"S range   :  [{S.min():.4f}, {S.max():.4f}]  (max possible 1/e ≈ 0.3679)")
print()

# ─── 5. Custom loss: MSE + 1e7 penalty for g ∉ [0,1] ────────────────────────
LOSS_G = """
function loss(prediction, target)
    mse = (prediction - target)^2
    lo  = max(zero(prediction), -prediction)^2
    hi  = max(zero(prediction), prediction - one(prediction))^2
    return mse + 1.0e7 * (lo + hi)
end
"""

model_g = PySRRegressor(
    binary_operators   = ["+", "-", "*", "/"],
    unary_operators    = ["exp", "log", "sqrt", "square"],
    elementwise_loss   = LOSS_G,
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

VAR_NAMES = ["lam", "mu", "gam"]   # "lambda"/"gamma" are reserved in sympy

print("=" * 60)
print("PySR  →  g = p₁/S  ∈ [0,1]")
print(f"         {mask.sum()} samples, mu ∈ {{50,100,125,150,175,200}}")
print("=" * 60)

model_g.fit(X, g_data, variable_names=VAR_NAMES)

print("\nPareto front  (g = p₁/S):")
print(model_g.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest expression for g : {model_g.sympy()}")

# ─── 6. Derive p₁ and p₂ ─────────────────────────────────────────────────────
pred_g  = model_g.predict(X)
pred_p1 = pred_g * S
pred_p2 = (1.0 - pred_g) * S

# ─── 7. Constraint verification ──────────────────────────────────────────────
r2 = lambda y, yh: 1 - np.var(y - yh) / np.var(y)

print()
print("=" * 60)
print("Constraint verification")
print("=" * 60)
print(f"g  ∈ [0,1]           : {np.all((pred_g  >= 0) & (pred_g  <= 1))}")
print(f"p₁ ∈ [0,S]           : {np.all((pred_p1 >= 0) & (pred_p1 <= S))}")
print(f"p₂ ∈ [0,S]           : {np.all((pred_p2 >= 0) & (pred_p2 <= S))}")
print(f"p₁ ∈ [0,1]           : {np.all((pred_p1 >= 0) & (pred_p1 <= 1))}")
print(f"p₂ ∈ [0,1]           : {np.all((pred_p2 >= 0) & (pred_p2 <= 1))}")
print(f"max|p₁+p₂ − S|       : {np.abs(pred_p1+pred_p2-S).max():.2e}  (exact by construction)")
print()
print(f"R² for g              : {r2(g_data, pred_g):.6f}")
print(f"R² for p₁             : {r2(y1, pred_p1):.6f}")
print(f"R² for p₂             : {r2(y2, pred_p2):.6f}")
print()

# ─── 8. R² breakdown by mu ───────────────────────────────────────────────────
print("R² breakdown by mu:")
print(f"  {'mu':>5}  {'n':>5}  {'R²(g)':>8}  {'R²(p₁)':>8}  {'R²(p₂)':>8}")
for mu_v in sorted(np.unique(X[:,1])):
    m = X[:,1] == mu_v
    print(f"  {mu_v:5.0f}  {m.sum():5d}  "
          f"{r2(g_data[m], pred_g[m]):8.5f}  "
          f"{r2(y1[m], pred_p1[m]):8.5f}  "
          f"{r2(y2[m], pred_p2[m]):8.5f}")
print()

# ─── 9. Sample rows ──────────────────────────────────────────────────────────
print("Sample check (first 5 rows):")
print(f"{'λ':>6} {'μ':>5} {'γ':>5}  {'S':>7}  "
      f"{'p₁_true':>9} {'p₁_pred':>9}  "
      f"{'p₂_true':>9} {'p₂_pred':>9}  {'Σpred':>8}")
for i in range(5):
    print(f"{X[i,0]:6.1f} {X[i,1]:5.0f} {X[i,2]:5.2f}  {S[i]:7.4f}  "
          f"{y1[i]:9.5f} {pred_p1[i]:9.5f}  "
          f"{y2[i]:9.5f} {pred_p2[i]:9.5f}  "
          f"{pred_p1[i]+pred_p2[i]:8.6f}")
