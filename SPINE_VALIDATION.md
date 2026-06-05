# Validation Note: SBI Spine Probabilities vs. Simulated State Probabilities

**Author:** Analysis of `Matrix.pdf` (SBI / Response 2) and `April16.pdf` (chain diagram)
**Data:** `merged_all_infinite_buffer.csv` — 6994 simulation rows
**Script:** `validate_spine.py`
**Date:** June 2026

---

## Motivation

The loss-rate validation (`LOSS_RATE_VALIDATION.md`) showed the SBI loss-rate
formula fails (R² = −147 to −32 across variants). The natural follow-up: do the
underlying **pure-idle-spine probabilities** `P0, P00, P000, …` themselves match
the simulated data? If they did, the error would be isolated in the loss
identity. They do **not** — the error is upstream, in the spine anchor `P0`.

---

## 1. State index mapping (from `April16.pdf`, Figure 1)

States are binary strings ordered by layer; within each layer, ascending binary
order. The empty state ∧ is index 0, so the pure-idle spine `0^(n)` is the first
state of each layer block, at index **2ⁿ − 1**:

| state | ∧ | `0` | `00` | `000` | `0000` | `00000` |
|---|---|---|---|---|---|---|
| index | 0 | 1 | 3 | 7 | 15 | 31 |

Verified against the known analytical identities (all R² ≈ 1.0):

| Identity | R² |
|---|---|
| `idx0 = P∧ = e^(−λ/γ)` | 1.000000 |
| `idx1 + idx2 = S = (λ/γ)e^(−λ/γ)` | 0.999999 |
| `λP0 = 2γ(P00+P01)` (idx 1; 3,4) | 0.999999 |
| `λP00 = 3γ(P000+P001)` (idx 3; 7,8) | 0.999999 |

The mapping `1, 3, 7, 15, 31` is confirmed.

---

## 2. SBI spine probabilities do not match

Using the Response-2 spine formula
`P0 = λ/(µ+γ+(λ/2)(1−β₀(2)))·e^(−λ/γ)`, `P_{0^(n)} = P0·∏_{k=2}^n R_k`:

| state | idx | R²(SBI vs data) | data mean | SBI mean |
|---|---|---|---|---|
| `0` | 1 | **−2.00** | 0.0732 | 0.1145 |
| `00` | 3 | **−0.84** | 0.0312 | 0.0379 |
| `000` | 7 | **−0.28** | 0.0194 | 0.0110 |
| `0000` | 15 | **−0.18** | 0.0130 | 0.0031 |
| `00000` | 31 | **−0.14** | 0.0006 | 0.0008 |

Every spine state has **negative R²** — the SBI formula is a worse predictor than
the column mean.

---

## 3. Diagnosis: the anchor inverts the idle/active split

The discrepancy is systematic. For λ=1.5, µ=50:

| γ | data P0 (idx1) | SBI P0 | S·µ/(µ+γ) | S·γ/(µ+γ) |
|---|---|---|---|---|
| 2.5  | **0.31366** | 0.01547 | **0.31361** | 0.01568 |
| 10   | **0.10745** | 0.02129 | **0.10759** | 0.02152 |
| 25   | **0.03758** | 0.01869 | **0.03767** | 0.01884 |

The pattern is unambiguous (light-load subset, λ<µ, 3366 rows):

- **data** `P0 = S·µ/(µ+γ)` → R² = 0.993 (the idle state carries the *large* share)
- **SBI** `P0 = S·γ/(µ+γ)` → R² = 0.976 (the SBI formula returns the *small* share)

So the SBI anchor computes the **active-state probability P1** and labels it the
**idle-state probability P0** — the two are swapped, an error of factor µ/γ ≈ 20.

### Root cause in the derivation

`Matrix.pdf` Response 2 (p.8) derives

$$
P_1 = \frac{\mu + \gamma + \frac{\lambda}{2}(1-\beta_0(2))}{\gamma}\, P_0
\;\;\Rightarrow\;\; \frac{P_1}{P_0} \approx \frac{\mu}{\gamma} \gg 1,
$$

claiming the **active** state is far more probable than the **idle** state. But
the server is idle ≈ 97% of the time (utilisation ρ = λ/µ ≈ 0.03), so the true
relation is the reverse, `P0/P1 = µ/γ` — precisely the ratio `r = µ/γ`
independently validated in the ratio analysis (`run_pysr_ratio.py`). Solving
`P0 + P1 = S` with the inverted ratio places the small value on `P0`:

$$
\text{SBI: } P_0 = \frac{S\gamma}{\mu+2\gamma+\dots} \approx \frac{S\gamma}{\mu}
\qquad\text{vs.}\qquad
\text{data: } P_0 = \frac{S\mu}{\mu+\gamma}.
$$

---

## 4. Conclusion

1. The **state index mapping is correct**: spine indices `1, 3, 7, 15, 31` (= 2ⁿ−1),
   verified against four analytical identities at R² ≈ 1.0.

2. The **simulated spine is clean and well-defined** — e.g. the idle state
   `P0 = S·µ/(µ+γ)` matches to ≈ 5 decimals at light load.

3. The **SBI spine formula does not match the data** (R² < 0 at every depth). The
   anchor `P0` is wrong by a factor of µ/γ because the Response-2 derivation
   inverts the `P1/P0` ratio (p.8). The idle and active states are swapped.

**Implication.** The error is **not** confined to the loss-rate identity — it
originates in the spine probability derivation itself. The loss-rate formula then
inherits this error. Both the spine probabilities and the loss rate built on them
are incorrect, and the failure traces to the single inverted ratio on p.8.
