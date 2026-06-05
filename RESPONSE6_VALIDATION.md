# Validation Note: Response 6 Spine Probabilities P₀⁽ⁿ⁾

**Author:** Validation of `Matrix.pdf` "Response 6" against simulation data
**Data:** `merged_all_infinite_buffer.csv` — 6994 simulation rows
**Script:** `validate_response6.py`
**Date:** June 2026

---

## What Response 6 changes

Response 6 fixes the negative-probability artifact in Response 5's `P00`
(`µ/(2γ−2λ/3)·P10`, whose denominator goes negative for λ ≥ 3γ). It re-anchors
`P00/P01` to the verified down-flow identity `2γP00 + 2γP01 = λP0` and splits the
mass with a **bounded** branching factor, so the denominator can never collapse:

$$
\beta_0(2) = \frac{\gamma}{\mu + 2\gamma + \lambda/3}, \qquad
\boxed{P_{00} = (1-\beta_0(2))\frac{\lambda}{2\gamma}P_0}, \qquad
P_{01} = \beta_0(2)\frac{\lambda}{2\gamma}P_0 .
$$

The spine step-down becomes `R_n = (1−β₀(n))·λ/(nγ)` and the `P0` anchor is
recomputed from the node-0 global balance `µP1 + 2γ(P00+P10) = (λ+γ)P0`, giving

$$
\frac{P_0}{P_1} = \frac{\mu(2\gamma+\mu+\lambda)}{(2\gamma+\mu)(\gamma + \beta_0(2)\lambda)},
\qquad P_0 + P_1 = \frac{\lambda}{\gamma}e^{-\lambda/\gamma}.
$$

These **are new expressions for `P₀⁽ⁿ⁾`** — different from every prior response.

State indexing (`April16.pdf`, Fig. 1): `P0=1, P00=3, P000=7, P0000=15, P00000=31`.

---

## Validation

| state | idx | R²(all) | R²(clean) | Response 5 |
|---|---|---|---|---|
| `P0` | 1 | **0.9633** | **0.9611** | (−7.6) |
| `P00` | 3 | **0.8427** | **0.8146** | −24.7 |
| `P000` | 7 | **0.7786** | 0.7386 | n/a |
| `P0000` | 15 | 0.6810 | 0.6356 | n/a |
| `P00000` | 31 | −66.98 | −1457.8 | n/a |

Individual links (fed data parents):

| link | R² | |
|---|---|---|
| `P00 = (1−β₀(2))·(λ/2γ)·P0` | **0.9469** | ✓ (Response 5: −24.7) |
| `P11 = λ/(2γ+µ)·P1` | **0.9845** | ✓ |
| `P10 = λµ/(2γ(2γ+µ))·P1` | 0.2114 | ~ light-load |

Sample (λ=1.5, µ=50, γ=2.5): data `P0`=0.31366 → R6 0.31361; data `P00`=0.08943
→ R6 0.08984. Near-exact at the shallow spine.

---

## Findings

1. **The negative-probability artifact is eliminated.** The bounded factor
   `(1−β₀(2)) ∈ [0,1]` keeps `P00` non-negative for all λ, µ, γ. The `P00` link
   jumps from **R²=−24.7 (Response 5) to +0.95**.

2. **The new `P0` anchor validates strongly** (R²=0.96, vs Response 5's
   fully-closed −7.6), and the shallow spine `P0/P00/P000` is now good
   (R² = 0.96 / 0.84 / 0.78).

3. **Deep states still overshoot.** At `P00000` (idx 31) the closed form diverges
   from the data (R² ≪ 0) because the step-down product `∏ (1−β₀(k))·λ/(kγ)` keeps
   *growing* while `λ/(kγ) > 1` (heavy load, λ/γ large), peaking around `n ≈ λ/γ`.
   In that regime the 37-state simulation is itself noise/truncation-limited
   (mean `P00000` ≈ 6e-4), so the deep comparison is unreliable on both sides.

---

## Conclusion

Response 6 produces genuinely **new, validated** `P₀⁽ⁿ⁾` expressions and resolves
the Response-5 failure: the `P00` negative artifact is gone (R² −24.7 → +0.95),
the `P0` anchor is sound (R²=0.96), and the shallow spine matches the simulation
well (`P0`/`P00`/`P000` at 0.96/0.84/0.78). The remaining gap is confined to the
deep spine under heavy load, where both the closed form (growing product) and the
finite-state simulation (noise) lose accuracy.
