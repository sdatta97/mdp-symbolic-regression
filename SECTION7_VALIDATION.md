# Validation Note: Section 7 "Complete Response"

**Author:** Validation of `Matrix.pdf` Section 7 against simulation data
**Data:** `merged_all_infinite_buffer.csv` — 6994 simulation rows
**Script:** `validate_section7.py`
**Date:** June 2026

---

## What Section 7 is

Section 7 consolidates Responses 4–6 into a single self-contained pipeline and,
for the first time, closes the Layer-1 system with an explicit `χ` ratio so the
**entire loss rate** can be evaluated end-to-end:

$$
\beta_0(2) = \frac{\gamma}{\mu + 2\gamma + \lambda/3}, \qquad
K = \frac{\mu(2\gamma+\mu+\lambda)}{(2\gamma+\mu)(\gamma + \lambda\beta_0(2))}, \qquad
\chi = \frac{K}{1+K},
$$
$$
P_0 = \chi S,\quad P_1 = (1-\chi)S,\quad S = \tfrac{\lambda}{\gamma}e^{-\lambda/\gamma};
$$
$$
P_{11} = \tfrac{\lambda}{2\gamma+\mu}P_1,\ \
P_{10} = \tfrac{\lambda\mu}{2\gamma(2\gamma+\mu)}P_1,\ \
P_{00} = (1-\beta_0(2))\tfrac{\lambda}{2\gamma}P_0;
$$
$$
R_n = (1-\beta_0(n))\tfrac{\lambda}{n\gamma},\quad
\text{Loss Rate} = \lambda - \mu\Big(1 - P_\wedge - \textstyle\sum_{n\ge1} P_{0^{(n)}}\Big).
$$

---

## 1. State probabilities

| state | idx | R² |
|---|---|---|
| `P0` | 1 | **0.9610** |
| `P11` | 6 | **0.9179** |
| `P00` | 3 | **0.8383** |
| `P000` | 7 | 0.7740 |
| `P0000` | 15 | 0.6761 |
| `P10` | 5 | 0.6645 |

Consistent with Response 6 (the truncated `β₀(2)` vs full recurrence barely
moves the anchor). The shallow spine and `P11` validate well; `P10` and the deep
spine are weaker.

---

## 2. Loss rate — the headline

For the **first time across all responses**, the loss rate comes out **positive
and in the right ballpark**:

| regime | R² |
|---|---|
| **overall** | **0.7271** |
| light load (λ<µ) | −0.3538 |
| heavy load (λ≥µ) | 0.7848 |
| competing-exponential `γ/(µ+γ)` (reference) | 0.8437 |

Prior responses gave **R² = −147 to −32 with unphysical negative loss rates**.
The χ-anchored spine fixes this: `1 − P∧ − Σspine` now tracks the true
server-busy probability (e.g. λ=1.5,µ=50,γ=2.5: 0.0278 vs true 0.0286).

By µ:

| µ | R² | mean abs err |
|---|---|---|
| 50 | **0.8020** | 0.050 |
| 100 | −0.384 | 0.190 |
| 125 | **0.8479** | 0.061 |
| 150 | −0.394 | 0.134 |
| 175 | −0.349 | 0.194 |
| 200 | −0.627 | 0.228 |

Sample (λ=1.5, µ=50): sim 0.0478 → S7 0.0748; γ=25: sim 0.3350 → S7 0.3433.

---

## 3. Assessment

**Major progress.** Section 7 is the first version where:
- the state probabilities validate well (`P0` R²=0.96, `P00` 0.84, `P11` 0.92), and
- the **loss rate is finally positive and physically sensible** (overall R²=0.73,
  up from −147…−32), because the χ-anchored spine now captures roughly the right
  idle mass.

**Remaining gap.** The overall positive R² is carried by the gross load trend
across the wide parameter sweep. Within homogeneous subsets the loss rate
**systematically overshoots**: per-µ it is positive only for µ∈{50,125} and
negative for µ∈{100,150,175,200} (mean abs error up to 0.23), and the light-load
subset is negative. The simple competing-exponential law `γ/(µ+γ)` (R²=0.84) is
still the more accurate closed form for the loss fraction.

The residual error traces to the loss identity's reliance on the **pure** idle
spine: `1 − P∧ − Σ_{pure spine}` equals the active mass only when the mixed-idle
states (`01`, `001`, …) are negligible. That holds at light load with small γ
(where the pure spine ≈ all idle mass) but breaks as load/γ grow, which is
exactly where Section 7's loss rate degrades.

---

## Conclusion

Section 7 is a clear step forward: the consolidated χ-anchored derivation makes
the spine probabilities accurate (`P0`/`P00`/`P11` ≈ 0.96/0.84/0.92) and, for the
first time, yields a **positive loss rate in the correct ballpark** (R²=0.73,
versus the negative/unphysical values of every earlier response). It is not yet
quantitatively exact — it overshoots systematically per-µ and trails the
empirical `γ/(µ+γ)` law — because the loss identity still counts only the pure
idle spine, omitting the mixed-idle mass that matters under heavier load.
