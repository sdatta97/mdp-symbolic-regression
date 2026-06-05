"""
Validate the Matrix.pdf (SBI) loss-rate formula against simulated loss rates.

PDF formula (Scalar Boundary Induction):
  P∧        = exp(−λ/γ)                                   empty-queue boundary
  β₀(n)     = γ / (µ + nγ + (λ/(n+1))·(1 − β₀(n+1)))      branching recurrence
  R_n       = λ / (nγ + λ·(1 − β₀(n)))                    spine multiplier
  P_{0^(n)} = P∧ · ∏_{k=1}^n R_k                          idle-spine probs
  LossRate  = λ − µ·(1 − Σ_{n=1}^∞ P_{0^(n)})            absolute loss rate

Simulated column avg_packet_loss_rate ∈ [0,1] is a *fraction*, so compare to
  loss_fraction = LossRate / λ = 1 − (µ/λ)·(1 − Σ_{n≥1} P_{0^(n)})
"""

import numpy as np
import pandas as pd

NMAX = 4000      # truncation depth for the infinite recurrence / sum

def sbi_loss_rate(lam, mu, gam, Nmax=NMAX):
    """Compute absolute loss rate and idle-spine sum via SBI recurrence."""
    # --- β₀(n) backward recurrence: start at tail limit γ/(µ+Nγ) ---
    n = Nmax
    beta_next = gam / (mu + n * gam)          # tail limit β₀(N+1)
    beta = np.empty(Nmax + 2)
    beta[Nmax + 1] = beta_next
    for n in range(Nmax, 0, -1):
        beta[n] = gam / (mu + n * gam + (lam / (n + 1)) * (1.0 - beta[n + 1]))

    # --- R_n and idle-spine probabilities ---
    Pwedge = np.exp(-lam / gam)
    prod = 1.0
    spine_sum = 0.0                            # Σ_{n≥1} P_{0^(n)}
    for n in range(1, Nmax + 1):
        Rn = lam / (n * gam + lam * (1.0 - beta[n]))
        prod *= Rn
        term = Pwedge * prod
        spine_sum += term
        if term < 1e-16 and n > 5:             # converged
            break

    loss_rate = lam - mu * (1.0 - spine_sum)
    return loss_rate, spine_sum, Pwedge


# ─── Load combined CSV ───────────────────────────────────────────────────────
df = pd.read_csv("/Users/sdatta/Downloads/Claude/merged_all_infinite_buffer.csv")
df = df[pd.to_numeric(df["lambda"], errors="coerce").notna()].copy()
for c in ("lambda", "mu", "gamma"):
    df[c] = df[c].astype(float)
df["avg_packet_loss_rate"] = pd.to_numeric(df["avg_packet_loss_rate"], errors="coerce")
df = df.dropna(subset=["avg_packet_loss_rate"]).reset_index(drop=True)

print(f"Rows: {len(df)}")

# ─── Compute SBI predictions ─────────────────────────────────────────────────
pred_frac = np.empty(len(df))
spine     = np.empty(len(df))
for i, row in df.iterrows():
    lr, ss, pw = sbi_loss_rate(row["lambda"], row["mu"], row["gamma"])
    pred_frac[i] = lr / row["lambda"]          # normalised loss fraction
    spine[i]     = ss

sim = df["avg_packet_loss_rate"].values

r2 = lambda y, yh: 1 - np.sum((y - yh) ** 2) / np.sum((y - np.mean(y)) ** 2)

mask = np.isfinite(pred_frac)
print(f"\nFinite predictions: {mask.sum()} / {len(df)}")
print()
print("=" * 60)
print("SBI loss-fraction  vs  simulated avg_packet_loss_rate")
print("=" * 60)
print(f"  R²            : {r2(sim[mask], pred_frac[mask]):.6f}")
print(f"  max |err|     : {np.abs(sim[mask]-pred_frac[mask]).max():.4e}")
print(f"  mean |err|    : {np.abs(sim[mask]-pred_frac[mask]).mean():.4e}")
print(f"  RMSE          : {np.sqrt(np.mean((sim[mask]-pred_frac[mask])**2)):.4e}")

print()
print("R² breakdown by μ:")
print(f"  {'μ':>5}  {'n':>5}  {'R²':>10}  {'maxErr':>10}  {'meanErr':>10}")
for mu_v in sorted(df["mu"].unique()):
    m = (df["mu"].values == mu_v) & mask
    if m.sum() < 2: continue
    print(f"  {int(mu_v):5d}  {m.sum():5d}  "
          f"{r2(sim[m], pred_frac[m]):10.6f}  "
          f"{np.abs(sim[m]-pred_frac[m]).max():10.4e}  "
          f"{np.abs(sim[m]-pred_frac[m]).mean():10.4e}")

# ─── Sample table ────────────────────────────────────────────────────────────
print()
print("Sample comparison (first 12 rows):")
print(f"{'λ':>6} {'μ':>5} {'γ':>6}  {'sim_loss':>9}  {'sbi_loss':>9}  "
      f"{'abs_err':>9}  {'Σspine':>8}")
for i in range(min(12, len(df))):
    print(f"{df['lambda'][i]:6.1f} {df['mu'][i]:5.0f} {df['gamma'][i]:6.2f}  "
          f"{sim[i]:9.5f}  {pred_frac[i]:9.5f}  "
          f"{abs(sim[i]-pred_frac[i]):9.2e}  {spine[i]:8.5f}")
