# Verification Note: Doc11.docx "Exact Match" Claim

**Author:** Cross-check of `Doc11.docx` against the validation in this repo
**Data:** `merged_all_infinite_buffer.csv` (6994 rows) + independent recomputation
**Script:** `verify_doc11.py`
**Date:** June 2026

---

## What Doc11 claims

Doc11 tabulates 4 scenarios with a near-perfect match (Δ ≈ 10⁻⁷) between
"Empirical Simulation" and a "Pure Scalar Formula":

| Scenario | λ | µ | γ | Sim | Formula | Δ |
|---|---|---|---|---|---|---|
| 1 High Load | 3.5 | 0.5 | 1.8 | 3.25055104 | 3.25055097 | 7e-8 |
| 2 Balanced | 1.5 | 1.5 | 1.0 | 0.78121045 | 0.78121021 | 2e-7 |
| 3 High Service | 1.2 | 2.0 | 1.0 | 0.65471481 | 0.65471466 | 1e-7 |
| 4 Low Load | 0.5 | 2.5 | 2.0 | 0.11903340 | 0.11903337 | 3e-8 |

This appears to contradict my validation (loss-rate R² ≈ 0.73, systematic
overshoot per-µ). It does **not**, for three independent reasons.

---

## 1. Different parameter regime — not in the validation data

Doc11 uses **µ ∈ {0.5, 1.5, 2.0, 2.5}**. The validation dataset
`merged_all_infinite_buffer.csv` has **µ ∈ {50, 100, 125, 150, 175, 200}**.

> Exact matches of Doc11 scenarios in the CSV: **0 of 4**.
> Rows with µ ≤ 2.5 in the CSV: **0**.

So Doc11's scenarios cannot be cross-checked against the simulation data my
validation used. My conclusions are stated for the µ = 50–200 regime; Doc11 is a
disjoint regime.

---

## 2. The genuine scalar formula does NOT reproduce Doc11's "Formula" column

I recomputed the **Section 7 scalar formula** (verified correct — it reproduces
the PDF's own worked example λ=1.5,µ=2,γ=1 → P0=0.2254, P1=0.1093 exactly) for
Doc11's four parameter sets:

| Scenario | Doc11 "Formula" | Section 7 scalar formula (mine) | difference |
|---|---|---|---|
| 1 | 3.250551 | 3.139510 | **0.111** |
| 2 | 0.781210 | 0.906209 | **0.125** |
| 3 | 0.654715 | 0.606806 | **0.048** |
| 4 | 0.119033 | 0.242845 | **0.124** |

The genuine scalar β-formula differs from Doc11's "Pure Scalar Formula" by
0.05–0.13. **Doc11's column is therefore not the independent scalar formula** —
it is being produced some other way (see §3).

---

## 3. The 10⁻⁷ agreement is a conservation tautology

The loss rate can be written two ways from the **same** state vector:

$$
\text{active side: } \lambda - \mu\!\!\sum_{x\in S_{\text{active}}}\!\!P_x
\qquad
\text{idle side: } \lambda - \mu\Big(1 - P_\wedge - \!\!\sum_{x\in S_{\text{idle}}}\!\!P_x\Big).
$$

Because the state space partitions as `empty ∪ idle ∪ active` and sums to 1,
these are **identically equal for any consistent probability vector** — the
agreement only confirms the partition, not the correctness of any scalar formula.
This is precisely the artifact the PDF's own **Response 3 (p.12)** already
confessed to:

> *"the script was verifying the global partition of the state space, not the
> individual correctness of the boundary equations… Because it pulled the correct
> values of P0 and P00 straight out of the solved matrix rather than computing
> them from my incomplete scalar formulas, the two sides were forced to be
> identical down to the last decimal place."*

**The crucial distinction** — `S_idle` must be **all** states starting with 0,
not the pure spine `{0,00,000,…}`. Demonstrated on the CSV (row λ=1.5,µ=50,γ=2.5):

| quantity | value |
|---|---|
| `λ − µ·Σ_active` | 0.431 |
| `λ − µ(1 − P∧ − Σ_ALL_idle)` | 0.435 (≈ identical; residual = 37-state truncation + MC noise) |
| `λ − µ(1 − P∧ − Σ_PURE_spine)` ← **the scalar formula uses this** | **0.060** |

Mean `Σ_pure_spine = 0.137` vs `Σ_all_idle = 0.400` over 6994 rows. The scalar
formula sums only the pure spine, so it gives a **different** answer (0.060 vs
0.431). Doc11's "Formula" column matching "Simulation" to 1e-7 is only possible
if it uses `Σ_all_idle` (or matrix-extracted state values), i.e. the tautology —
not the pure-spine scalar formula.

---

## Conclusion

**My validation is correct.** It is an *independent* test: the scalar formula is
computed from the β₀-recurrence (never from the matrix/simulation) and compared
against independent Monte-Carlo simulation data. In that test the formula is
imperfect (R² ≈ 0.73, systematic per-µ overshoot) in the µ = 50–200 regime.

**Doc11 does not refute this**, because:

1. Its scenarios (µ = 0.5–2.5) are outside the validation dataset entirely.
2. The genuine scalar formula, applied to Doc11's parameters, differs from
   Doc11's "Formula" column by 0.05–0.13 — so that column is not the scalar
   formula.
3. The 10⁻⁷ "match" is the conservation-partition tautology
   (`Σ_active ≡ 1 − P∧ − Σ_all_idle`), the exact artifact the PDF's Response 3
   already identified — it confirms the state vector sums to 1, not that the
   scalar formula is exact.

A genuine validation of Doc11's claim would require (a) running the Monte-Carlo
simulation for those small-µ parameters and (b) computing the scalar formula
**independently** of that simulation, then comparing. Re-using matrix/simulation
state values on both sides cannot test the formula.
