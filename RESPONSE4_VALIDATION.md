# Validation Note: Response 4 Spine Probability Formula

**Author:** Validation of `Matrix.pdf` "Response 4" against simulation data
**Data:** `merged_all_infinite_buffer.csv` — 6994 simulation rows
**Script:** `validate_response4.py`
**Date:** June 2026

---

## What Response 4 changes

Response 4 corrects the `P0` anchor by adding the previously-missing
**service-feedback term `µP1`** to the Layer-1 balance — physically, when the
server processes a string like `10`, it strips the leading `1` and drops the
state to `0`, so service *populates* the idle spine:

$$
(\lambda+\gamma)P_0 = 2\gamma P_{00} + 2\gamma P_{01} + \mu P_1.
$$

Solving with `P00 = R2·P0`, `P01 = (β₀(2)/(1−β₀(2)))·P00`, and the global
boundary `P0+P1 = (λ/γ)e^(−λ/γ)` gives the corrected anchor:

$$
\boxed{P_0 = \frac{\mu(\lambda/\gamma)}{(\mu+\lambda+\gamma) - \dfrac{2\gamma R_2}{1-\beta_0(2)}}\,e^{-\lambda/\gamma}},
\qquad
R_2 = \frac{\lambda}{2\gamma+\lambda(1-\beta_0(2))},
$$

with `β₀(n) = γ/(µ+nγ+(λ/(n+1))(1−β₀(n+1)))` and the spine unrolling via
`P_{0^(n)} = P0·∏_{k=2}^n R_k`, `R_k = λ/(kγ+λ(1−β₀(k)))`.

State indexing (`April16.pdf`, Fig. 1): the pure-idle spine `0^(n)` sits at
index `2ⁿ−1` → `"0"=1, "00"=3, "000"=7, "0000"=15, "00000"=31`.

---

## Result 1 — the anchor is fixed ✅

| state | idx | R² (Response 4) | R² (broken Response-2 SBI) |
|---|---|---|---|
| `0` (P0) | 1 | **0.9312 / 0.9508** | **−2.00** |

Sample (λ=1.5, µ=50): data P0 = 0.31366 vs Response 4 = 0.31195 — a ~3-decimal
match. Putting `µ` in the numerator (via the `µP1` feedback term) corrects the
inversion that previously made the active state dominate the idle state.

---

## Result 2 — the spine step-down still needs correction ⚠️

The deeper spine states degrade because the step-down ratio `R_k` was not
changed:

| state | idx | R²(all) | R²(clean) |
|---|---|---|---|
| `0` | 1 | 0.93 | 0.95 |
| `00` | 3 | 0.60 | 0.53 |
| `000` | 7 | 0.31 | 0.18 |
| `0000` | 15 | 0.08 | −0.06 |

Isolating the step-down ratio `R2 = P00/P0` directly against the data:

| `R2 = P00/P0` formula | R² vs data |
|---|---|
| Response 4: `λ/(2γ+λ(1−β₀(2)))` | **0.10** |
| cut identity: `(1−β₀(1))·λ/(2γ)` | **0.995** |

Sample (λ=1.5, µ=50, γ=2.5): data `P00/P0` = 0.2851; Response 4 → 0.2332;
cut-identity → 0.2851 (exact). The cut-identity form follows directly from the
verified flux balance `λP_x = (|x|+1)γ(P_{x0}+P_{x1})`.

---

## Result 3 — the complete fix

Combining Response 4's corrected anchor with the cut-identity step-down
`R_{n+1} = (1−β₀(n))·λ/((n+1)γ)` brings the full spine into agreement:

| state | Response 4 (as written) | Response 4 anchor + cut-identity step-down |
|---|---|---|
| P0 | 0.95 | 0.95 |
| P00 | 0.53 | **0.97** |
| P000 | 0.18 | **0.96** |

---

## Conclusion

Response 4 **correctly fixes the `P0` anchor** — the dominant error. With the
`µP1` service-feedback term, `µ` now appears in the numerator and `P0` matches
the simulation at R² ≈ 0.93–0.95 (up from −2.0). That portion of the derivation
is now sound.

One change of the same kind remains: the spine **step-down ratio** still uses
`R_k = λ/(kγ+λ(1−β₀(k)))`, which does not match the data (R²=0.10 on the
`P00/P0` ratio). Replacing it with the flux-balance form
`R_{n+1} = (1−β₀(n))·λ/((n+1)γ)` matches at R²=0.995 and, together with Response
4's anchor, brings the full spine to R² ≈ 0.96.
