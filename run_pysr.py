"""
PySR — independent fits for f₁ and f₂ with sum constraint

Target form:
    p₁ = f₁(λ,μ,γ) · e^(−λ/γ)
    p₂ = f₂(λ,μ,γ) · e^(−λ/γ)
    f₁ + f₂ = λ/γ                  (sum constraint)
    0 ≤ f₁ ≤ λ/γ,  0 ≤ f₂ ≤ λ/γ   (bound constraints)

Two-pass independent fitting
─────────────────────────────
Pass 1 — fit f₁:
    target  = y₁·e^(λ/γ)
    weights = ρ = λ/γ  (upper bound in loss)
    loss    = MSE + 1e7·[f₁<0]² + 1e7·[f₁>ρ]²

Pass 2 — fit f₂ independently, sum-constrained:
    target  = y₂·e^(λ/γ)
    weights = ρ − f₁_pred  (= ideal f₂ for exact sum)
    loss    = MSE + 1e7·(f₂ − weight)²   ← heavy sum-constraint penalty
                  + 1e7·[f₂<0]²          ← non-negativity
"""

import re
import numpy as np
import pandas as pd
from pysr import PySRRegressor

# ─── 1. Load & clean ─────────────────────────────────────────────────────────
df = pd.read_csv("/Users/sdatta/Downloads/Claude/merged_infinite_buffer_losses.csv")
df = df[pd.to_numeric(df["lambda"], errors="coerce").notna()].copy()
for col in ("lambda", "mu", "gamma"):
    df[col] = df[col].astype(float)

# ─── 2. Parse state probabilities (mean over 10 runs) ────────────────────────
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

# ─── 3. Derived targets ──────────────────────────────────────────────────────
# f₁ = p₁·e^(λ/γ),  f₂ = p₂·e^(λ/γ),  f₁+f₂ = λ/γ
exp_pos  = np.exp(lam_all / gam_all)           # e^(+λ/γ)
rho_all  = lam_all / gam_all                   # λ/γ — the sum target
S_all    = (lam_all / gam_all) * np.exp(-lam_all / gam_all)
f1_all   = y1_all * exp_pos
f2_all   = y2_all * exp_pos

# ─── 4. Filter to S > 0.001 (noise-safe region, same as before) ──────────────
SMIN = 1e-3
mask = S_all > SMIN

X    = np.column_stack([lam_all[mask], mu_all[mask], gam_all[mask]])
rho  = rho_all[mask]
f1   = f1_all[mask]
f2   = f2_all[mask]
y1   = y1_all[mask]
y2   = y2_all[mask]
lam  = lam_all[mask]
gam  = gam_all[mask]

print(f"Training rows : {mask.sum()} / {len(df)}")
print(f"ρ = λ/γ range : [{rho.min():.4f}, {rho.max():.4f}]")
print(f"f₁ range      : [{f1.min():.4f}, {f1.max():.4f}]")
print(f"f₂ range      : [{f2.min():.6f}, {f2.max():.4f}]")
print(f"Sum check (data)  max|f₁+f₂ − ρ| : {np.abs(f1+f2-rho).max():.3e}")
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
# PASS 1 — fit f₁  (bounds: 0 ≤ f₁ ≤ ρ)
# ═══════════════════════════════════════════════════════════════════════════════
LOSS_F1 = """
function loss(prediction, target, weight)
    # weight = ρ = λ/γ
    mse = (prediction - target)^2
    lo  = max(zero(prediction), -prediction)^2
    hi  = max(zero(prediction), prediction - weight)^2
    return mse + 1.0e7 * (lo + hi)
end
"""

print("=" * 60)
print("Pass 1  →  fitting  f₁  (target = p₁·e^(λ/γ), bound [0, ρ])")
print("=" * 60)

model_f1 = PySRRegressor(elementwise_loss=LOSS_F1, **COMMON)
model_f1.fit(X, f1, weights=rho, variable_names=VAR_NAMES)

print("\nPareto front (f₁):")
print(model_f1.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest f₁ expression : {model_f1.sympy()}")

f1_pred = model_f1.predict(X)

# ═══════════════════════════════════════════════════════════════════════════════
# PASS 2 — fit f₂  independently, with sum constraint
# weights = ρ − f₁_pred  (ideal f₂ for exact sum)
# loss enforces:  f₂ ≈ ρ − f₁_pred  (sum constraint, 1e7 penalty)
#                 f₂ ≈ true data     (MSE term)
#                 f₂ ≥ 0             (non-negativity, 1e7 penalty)
# ═══════════════════════════════════════════════════════════════════════════════
LOSS_F2 = """
function loss(prediction, target, weight)
    # weight = ρ − f₁_pred  (sum-consistent ideal value for f₂)
    mse     = (prediction - target)^2
    sum_pen = (prediction - weight)^2        # enforce f₂ = ρ − f₁_pred
    lo      = max(zero(prediction), -prediction)^2
    return mse + 1.0e7 * sum_pen + 1.0e7 * lo
end
"""

f2_ideal = rho - f1_pred   # per-sample ideal f₂ for exact sum

print()
print("=" * 60)
print("Pass 2  →  fitting  f₂  (sum-constrained against f₁ predictions)")
print("=" * 60)

model_f2 = PySRRegressor(elementwise_loss=LOSS_F2, **COMMON)
model_f2.fit(X, f2, weights=f2_ideal, variable_names=VAR_NAMES)

print("\nPareto front (f₂):")
print(model_f2.equations_[["complexity", "loss", "equation"]].to_string())
print(f"\nBest f₂ expression : {model_f2.sympy()}")

f2_pred = model_f2.predict(X)

# ─── Recover p₁, p₂ ──────────────────────────────────────────────────────────
exp_neg  = np.exp(-lam / gam)
pred_p1  = f1_pred * exp_neg
pred_p2  = f2_pred * exp_neg

# ═══════════════════════════════════════════════════════════════════════════════
# Constraint verification
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Constraint verification")
print("=" * 60)
print(f"f₁ ∈ [0, ρ]          : {np.all((f1_pred >= 0) & (f1_pred <= rho))}")
print(f"f₂ ∈ [0, ρ]          : {np.all((f2_pred >= 0) & (f2_pred <= rho))}")
print(f"p₁ ∈ [0, 1]          : {np.all((pred_p1 >= 0) & (pred_p1 <= 1))}")
print(f"p₂ ∈ [0, 1]          : {np.all((pred_p2 >= 0) & (pred_p2 <= 1))}")
print()
sum_err = np.abs(f1_pred + f2_pred - rho)
print(f"Sum |f₁+f₂ − ρ|  max : {sum_err.max():.4e}")
print(f"Sum |f₁+f₂ − ρ| mean : {sum_err.mean():.4e}")
print()
print(f"R²  f₁  : {r2(f1, f1_pred):.6f}")
print(f"R²  f₂  : {r2(f2, f2_pred):.6f}")
print(f"R²  p₁  : {r2(y1, pred_p1):.6f}")
print(f"R²  p₂  : {r2(y2, pred_p2):.6f}")
print()

# ─── Sample check ─────────────────────────────────────────────────────────────
print("Sample check (first 5 rows):")
print(f"{'λ':>5} {'γ':>5}  {'ρ=λ/γ':>6}  "
      f"{'f₁_true':>9} {'f₁_pred':>9}  "
      f"{'f₂_true':>9} {'f₂_pred':>9}  "
      f"{'Σ pred':>8}  {'ρ':>8}")
for i in range(5):
    print(f"{X[i,0]:5.2f} {X[i,2]:5.2f}  {rho[i]:6.4f}  "
          f"{f1[i]:9.5f} {f1_pred[i]:9.5f}  "
          f"{f2[i]:9.5f} {f2_pred[i]:9.5f}  "
          f"{f1_pred[i]+f2_pred[i]:8.6f}  {rho[i]:8.6f}")
