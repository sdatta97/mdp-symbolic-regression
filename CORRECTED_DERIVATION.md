# Corrected Derivation of the Pure-Idle-Spine Probabilities

**Author:** Re-derivation fixing the errors in `Matrix.pdf` (SBI / Response 2)
**Data:** `merged_all_infinite_buffer.csv` — 6994 simulation rows
**Script:** `derive_corrected_spine.py`
**Date:** June 2026

This note works the spine derivation from the **verified** partial-balance
equations of `April16.pdf` (the ones confirmed at R² ≈ 1.0), locates the two
errors in the SBI route, and gives corrected formulas that validate against the
data at R² ≈ 0.999.

---

## Setup (verified facts from `April16.pdf`)

State indexing (Fig. 1): binary strings by layer, ascending within layer; the
pure-idle spine `0^(n)` sits at index `2ⁿ − 1` → `"0"=1, "00"=3, "000"=7, …`.

Verified relations (R² ≈ 1.0 against data):

$$
P_\wedge = e^{-\lambda/\gamma}, \qquad
P_0 + P_1 = S \equiv \tfrac{\lambda}{\gamma}e^{-\lambda/\gamma}, \qquad
\boxed{\lambda P_x = (|x|+1)\,\gamma\,(P_{x0} + P_{x1})}\ \text{(cut identity)}
$$

plus the partial-balance equations (1.1)–(2.2):

$$
\lambda P_\wedge + \gamma P_{01} = (\mu+\gamma)P_1, \qquad
\mu P_1 + \gamma P_{00} = (\tfrac{\lambda}{2}+\gamma)P_0, \qquad
2\gamma P_{10} + \gamma P_{00} = \tfrac{\lambda}{2}P_0, \qquad
P_{01} = 2P_{10}.
$$

---

## Error 1 — the anchor inverts the µ/γ ratio

**Derivation.** From (1.1) with `λP∧ = γ(P0+P1)`:

$$
\gamma(P_0+P_1) + \gamma P_{01} = (\mu+\gamma)P_1
\;\Rightarrow\;
\gamma P_0 + \gamma P_{01} = \mu P_1
\;\Rightarrow\;
\boxed{P_1 = \frac{\gamma}{\mu}\,(P_0 + P_{01})}.
$$

So `P1/P0 ≈ γ/µ ≪ 1` — the **idle** state dominates. Solving `P0 + P1 = S`:

$$
\boxed{P_0 = \frac{S\,\mu}{\mu + \gamma(1+b)}},\qquad b = \frac{P_{01}}{P_0},
\qquad\text{leading order: } P_0 \approx \frac{S\mu}{\mu+\gamma}.
$$

**The PDF error.** `Matrix.pdf` (p.8) instead writes
`P1 = [(µ+γ+(λ/2)(1−β₀(2)))/γ]·P0`, i.e. `P1/P0 ≈ µ/γ ≫ 1` — claiming the
**active** state dominates. This is the reciprocal of the truth; it puts the
small share on `P0`, giving `P0 = Sγ/(µ+…)` (γ in the numerator).

**Validation.**

| relation | R² vs data |
|---|---|
| `P1 = (γ/µ)(P0+P01)` (corrected) | **0.9938** |
| `P1 = (µ+γ)/γ · P0` (PDF) | **−75.4** |
| `P0 = S·µ/(µ+γ(1+b))` (corrected) | **0.9995** |
| `P0 ≈ S·µ/(µ+γ)` (leading order) | 0.9705 |
| `P0 = S·γ/(µ+γ)` (PDF) | **−3.23** |

---

## Error 2 — the spine step-down multiplier

**Derivation.** The cut identity at `x = 0^(n)` (length `n`) gives
`P_{0^n0} + P_{0^n1} = λP_{0^n}/((n+1)γ)`. With the branching fraction
`β₀(n) = P_{0^n1}/(P_{0^n0}+P_{0^n1})` and `P_{0^(n+1)} = P_{0^n0}`:

$$
\boxed{R_{n+1} = \frac{P_{0^{(n+1)}}}{P_{0^{(n)}}} = \big(1-\beta_0(n)\big)\frac{\lambda}{(n+1)\gamma}}.
$$

**The PDF error.** `Matrix.pdf` uses `R_n = λ/(nγ + λ(1−β₀(n)))` — a different
denominator structure not implied by the cut identity.

**Validation** (clean rows, depth-3 spine above the noise floor):

| relation | R² vs data |
|---|---|
| `P00 = (1−β₀(1))·λ/(2γ)·P0` (corrected) | **0.999999** |
| `P000 = (1−β₀(2))·λ/(3γ)·P00` (corrected) | **0.999999** |
| `P00 = λ/(2γ+λ(1−β₀(1)))·P0` (PDF) | **0.664** |

---

## Corrected closed form

Unrolling the recursion from the anchor:

$$
\boxed{\,P_{0^{(n)}} = \frac{\mu}{\mu+\gamma}\cdot\frac{(\lambda/\gamma)^n\,e^{-\lambda/\gamma}}{n!}\cdot\prod_{k=1}^{n-1}\big(1-\beta_0(k)\big)\,}
$$

The idle queue length follows a **Poisson(λ/γ)** law, scaled by the served
fraction `µ/(µ+γ)` and thinned by the branch-loss factors `(1−β₀(k))`.

**Validation of the full corrected spine** (anchor + cut recursion, measured β₀):

| state | idx | R² (corrected) | R² (broken SBI) |
|---|---|---|---|
| `0` | 1 | **0.9994** | −2.00 |
| `00` | 3 | **0.9996** | −0.84 |
| `000` | 7 | **0.9997** | −0.28 |

Leading-order Poisson form (`β₀→0`), no fitted quantities:

| state | R²(all) | R²(λ<µ) |
|---|---|---|
| `0` | 0.97 | 0.99 |
| `00` | 0.57 | 0.73 |
| `000` | 0.19 | 0.40 |

(The leading-order form omits the `∏(1−β₀)` thinning, so it slightly overshoots
at depth; including the branch factors recovers R² ≈ 0.999.)

---

## Summary of the fix

| | PDF (SBI / Response 2) | Corrected |
|---|---|---|
| Anchor | `P0 = Sγ/(µ+…)` (γ on top) | `P0 = Sµ/(µ+γ(1+b))` (µ on top) |
| `P1/P0` | `≈ µ/γ` (active dominates) | `≈ γ/µ` (idle dominates) |
| Spine `R` | `λ/(nγ+λ(1−β₀(n)))` | `(1−β₀(n))·λ/((n+1)γ)` |
| Spine law | (none clean) | `µ/(µ+γ)·Poisson(n;λ/γ)·∏(1−β₀)` |
| Spine R² | < 0 at every depth | ≈ 0.999 |

Both errors trace to a single conceptual slip: treating the service rate µ as
something that **enhances** the active-state population, when high service in
fact **suppresses** it (utilisation ρ = λ/µ ≪ 1, so the server is idle almost
always). Fixing the µ/γ ratio at the anchor, and taking the spine multiplier
directly from the verified cut identity, makes the spine match the simulation at
R² ≈ 0.999.

A correct loss-rate identity can now be built on this corrected spine — but note
that even a perfect pure-idle spine only accounts for part of the server-idle
probability (the rest sits in mixed-idle states `01, 001, …`), which is why the
empirical competing-exponential law `loss = γ/(µ+γ)` remains the most practical
closed form for the loss rate itself.
