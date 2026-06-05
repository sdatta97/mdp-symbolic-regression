# Validation Note: Response 5 Spine / Loss Formulas

**Author:** Validation of `Matrix.pdf` "Response 5" against simulation data
**Data:** `merged_all_infinite_buffer.csv` — 6994 simulation rows
**Script:** `validate_response5.py`
**Date:** June 2026

---

## What Response 5 changes

Response 5 abandons the β₀ continued-fraction recurrence and instead takes the
spine step-down ratio **directly from the verified down-flow identity**
`λP_x = |x|γ(P_{x0}+P_{x1})`:

$$
R_n = \frac{P_{0^{(n)}}}{P_{0^{(n-1)}}} = \frac{\lambda}{n\gamma} - \xi_n,
\qquad \xi_n = \frac{P_{0^{(n-1)}1}}{P_{0^{(n-1)}}}.
$$

It also solves the Layer-1/Layer-2 states exactly via global balance at the
corner nodes, bypassing the partial-balance "guesses" (Eqs 1.1–2.2):

$$
P_{11} = \tfrac{\lambda}{2\gamma+\mu}P_1,\quad
P_{10} = \tfrac{\lambda\mu}{2\gamma(2\gamma+\mu)}P_1,\quad
P_{00} = \tfrac{\mu}{2\gamma-2\lambda/3}P_{10},
$$
$$
P_0 = \frac{\mu(\lambda/\gamma) + 2\gamma(P_{10}+P_{00})/P_\wedge}{\mu+\lambda+\gamma}\,e^{-\lambda/\gamma}.
$$

State indexing (`April16.pdf`, Fig. 1): `P∧=0, P0=1, P1=2, P00=3, P01=4, P10=5, P11=6`.

---

## 1. Validation as written (no corrections)

| Formula | R² vs data | Verdict |
|---|---|---|
| `P0` anchor (fed data `P10,P00`) | **1.0000** | ✅ exact structure |
| `R_n = λ/(nγ) − ξ_n` (n=2) | **0.9995** | ✅ structurally exact |
| `P11 = λ/(2γ+µ)·P1` | **0.9845** | ✅ strong |
| `P10 = λµ/(2γ(2γ+µ))·P1` | 0.2114 | ~ good at light load |
| `P00 = µ/(2γ−2λ/3)·P10` | **−24.71** | ❌ broken |
| Fully closed-form `P0` | **−7.63** | ❌ inherits P00 |

**Findings.**

- The **architecture is sound.** The `P0` anchor equation is *exact* (R²=1.0)
  when given the correct `P10, P00`, and the new step-down ratio
  `R_n = λ/(nγ) − ξ_n` is an exact identity (R²=0.9995). Moving off the
  partial-balance guesses onto the verified down-flow identity is the right call.
- `P11` validates strongly (R²=0.98).
- **The `P00` closed form is the one broken link.** Its denominator `2γ − 2λ/3`
  becomes **zero or negative whenever λ ≥ 3γ** (1916 of 6994 rows), producing
  diverging / negative probabilities. Because the anchor and the loss sum both
  depend on `P00`, this single error propagates into the fully closed-form `P0`
  (R²=−7.6) and the loss rate.

---

## 2. Proposed fix for the P00 link

The node-00 global balance in Response 5 has two slips:

1. It **omits the arrival inflow from the parent** `0 → 00` (rate `λ/2`).
2. It **adds a spurious outflow edge** `00 → 010` — but `010` is a child of `01`,
   not of `00` (the children of `00` are `000` and `001`). This bogus third
   arrival edge is what injects the `−2λ/3` into the denominator.

The correct global balance at `00` is

$$
\underbrace{\tfrac{\lambda}{2}P_0}_{\text{arrival from parent }0} + \underbrace{\mu P_{10}}_{\text{service }10\to00} + 3\gamma(P_{000}+P_{001})
= \underbrace{\lambda P_{00}}_{\text{arrivals out}} + \underbrace{2\gamma P_{00}}_{\text{decay to }0},
$$

and substituting the verified identity `λP_{00} = 3γ(P_{000}+P_{001})` collapses
it to a clean, pole-free closed form:

$$
\boxed{P_{00} = \frac{\tfrac{\lambda}{2}P_0 + \mu P_{10}}{2\gamma}}.
$$

| `P00` formula | R² vs data |
|---|---|
| Response 5: `µ/(2γ−2λ/3)·P10` | **−24.71** |
| Fix: `[(λ/2)P0 + µP10]/(2γ)` | **0.9573** |

Sample (λ=1.5, µ=50): data `P00`=0.08943, Response 5 → 0.05359, fix → 0.08992.

---

## 3. Self-contained spine with the fix

Solving the coupled Layer-1/Layer-2 system with the corrected `P00` (everything
in closed form, no data inputs):

| state | R² | (Response 5 as written) |
|---|---|---|
| `P0` | **0.9705** | −7.63 |
| `P00` | **0.8893** | −24.71 |
| `P10` | 0.6134 | 0.2114 |

---

## Conclusion

Response 5's **framework is correct**: the down-flow step-down ratio
`R_n = λ/(nγ) − ξ_n` is exact, the `P0` anchor equation is exact, and `P11`
validates at R²=0.98. The single defect is the **`P00` closed form**, whose
`2γ − 2λ/3` denominator diverges for `λ ≥ 3γ` due to a mis-counted arrival edge
at node `00`. Replacing it with the balance-correct
`P00 = [(λ/2)P0 + µP10]/(2γ)` removes the pole and lifts the self-contained spine
to `P0` R²=0.97, `P00` R²=0.89. The residual gap at depth is the usual
exact-but-deeper `ξ_k` tail, which is the natural next target.
