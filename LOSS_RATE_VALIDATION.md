# Validation Note: SBI Loss-Rate Formula vs. Simulated Loss Rates

**Author:** Analysis of `Matrix.pdf` (Scalar Boundary Induction proof)
**Data:** `merged_all_infinite_buffer.csv` — 6994 simulation rows, μ ∈ {50, 100, 125, 150, 175, 200}
**Script:** `validate_loss_rate.py`
**Date:** June 2026

---

## 1. The formula under test

`Matrix.pdf` derives a scalar (non-matrix) closed form for the steady-state loss
rate of the state-dependent queue via **Scalar Boundary Induction (SBI)**. The
headline identity is

$$
\text{Loss Rate} \;=\; \lambda - \mu\left(1 - \sum_{n=1}^{\infty} P_{0^{(n)}}\right)
$$

supported by the scalar spine machinery

$$
P_\wedge = e^{-\lambda/\gamma}, \qquad
\beta_0(n) = \frac{\gamma}{\mu + n\gamma + \frac{\lambda}{n+1}\left(1-\beta_0(n+1)\right)},
$$
$$
R_n = \frac{\lambda}{n\gamma + \lambda\left(1-\beta_0(n)\right)}, \qquad
P_{0^{(n)}} = P_\wedge \prod_{k=1}^{n} R_k .
$$

The PDF claims this reproduces the simulation metrics with **R² = 1.0**.

The simulated column `avg_packet_loss_rate` is a **fraction** in `[0, 1]`, so the
comparison target is the normalised loss fraction
`LossRate / λ = 1 − (μ/λ)(1 − Σ P_{0^(n)})`.

---

## 2. Result: the literal formula does not validate

The recurrence was implemented faithfully (backward iteration from depth
`N = 4000` at the tail limit `β₀ → γ/(μ+nγ)`; spine sum truncated at machine
precision). Compared against all 6994 rows:

| Model | R² vs `avg_packet_loss_rate` |
|---|---|
| **SBI literal formula** (Response 1, `λ − μ(1 − Σ)`) | **−147.2** ❌ |
| **SBI corrected** (`λ − μ(1 − P∧ − Σ)`, spine via `R₁`) | **−17.6** ❌ |
| **SBI Response 2** (`P₀ = λ/(μ+γ+(λ/2)(1−β₀(2)))·e^(−λ/γ)`) | **−32.0** ❌ |
| Competing exponentials `γ/(μ+γ)` | 0.844 |
| Load-aware `γ/(μ+γ) + μ/(μ+γ)·max(0, 1−μ/λ)` | **0.972** ✅ |

The SBI formula produces **unphysical negative loss rates**. Representative rows:

| λ | μ | γ | simulated | SBI formula |
|---|---|---|---|---|
| 1.5 | 50 | 2.5  | 0.0478 | **−23.41** |
| 1.5 | 50 | 10.0 | 0.1675 | **−28.23** |
| 1.5 | 50 | 70.0 | 0.5847 | **−31.63** |

### Why it fails

The term `μ(1 − Σ P_{0^(n)})` is intended to be the system throughput. With
μ = 50, λ = 1.5, the computed `1 − Σspine ≈ 0.73`, giving an implied throughput of

$$
50 \times 0.73 = 36.6 \;\gg\; \lambda = 1.5 .
$$

A throughput exceeding the arrival rate by **24×** is impossible — in steady state
throughput ≤ arrivals. The conceptual error:

> `Σ_{n≥1} P_{0^(n)}` sums only the **pure-zero spine** (states `0, 00, 000, …`).
> But `1 − (pure spine)` is **not** P(server busy). The server is idle on *every*
> string with head-bit 0 — including the empty state and all *mixed-idle* states
> (`01`, `001`, `010`, …). These are silently counted as "busy," inflating the
> throughput term far beyond λ.

For the identity to hold, `Σ` would need to capture ≈ 97% of probability mass
(the true server-idle fraction at utilisation ρ = λ/μ = 0.03); the pure spine
captures only ≈ 27%.

---

## 2a. Corrected variant: subtracting the empty state P∧

A natural fix is to exclude the empty-state mass from the throughput term as well:

$$
\text{Loss Rate} = \lambda - \mu\left(1 - P_\wedge - \sum_{n=1}^{\infty} P_{0^{(n)}}\right).
$$

This improves the fit by roughly 8× (overall **R² = −147 → −17.6**, mean abs error
1.20 → 0.45) but **still fails** — R² remains strongly negative in every μ group,
and low-γ rows still give negative loss rates (e.g. λ=1.5, μ=50, γ=2.5 → −5.11).
Worse, the correction overshoots at high γ:

| λ | μ | γ | `1−P∧−Σ` | simulated | corrected SBI |
|---|---|---|---|---|---|
| 1.5 | 50 | 2.5  | 0.183  | 0.0478 | **−5.11** (still negative) |
| 1.5 | 50 | 10.0 | 0.016  | 0.1675 | 0.464 (2.8× over) |
| 1.5 | 50 | 25.0 | 0.0023 | 0.3350 | 0.924 (over) |
| 1.5 | 50 | 70.0 | 0.0002 | 0.5847 | 0.994 (saturates) |

### Normalisation is not the cause

The comparison divides the absolute SBI loss rate by λ to match the simulated
*fraction* — i.e. `loss = 1 − (μ/λ)(1 − P∧ − Σ)`, the standard loss-system form.
This is the correct normalisation, and it is **not** what makes the formula fail.
A division by the positive constant λ cannot flip a negative absolute rate
positive, nor fix a γ-dependent error.

The decisive check compares the SBI busy-probability term `(1 − P∧ − Σ)` against
the **true, model-independent** server-busy probability, obtained purely from the
simulated loss via flow balance `throughput = μ·P(busy) = λ(1 − loss)`:

| λ | μ | γ | true P(busy) = λ(1−loss)/μ | SBI `1 − P∧ − Σ` | ratio |
|---|---|---|---|---|---|
| 1.5 | 50 | 2.5  | 0.0286 | 0.1834 | **6.4×** |
| 1.5 | 50 | 10.0 | 0.0250 | 0.0161 | 0.6× |
| 1.5 | 50 | 25.0 | 0.0200 | 0.0023 | **0.1×** |
| 1.5 | 50 | 70.0 | 0.0125 | 0.0002 | **0.02×** |

If these two columns matched, the formula would validate at *any* consistent
normalisation. Instead they diverge by 6× to 60×, and in **opposite directions**
as γ varies — confirming the failure is structural (the spine term is not
P(server busy)), independent of how the loss rate is scaled.

---

## 2b. Response 2 variant: closed-form spine anchor

A later revision of `Matrix.pdf` ("Response 2") keeps the corrected identity
`Loss Rate = λ − μ(1 − P∧ − Σ P_{0^(n)})` but replaces the spine anchor. Instead
of `P_{0^(1)} = P∧·R₁` with `R₁ = λ/(γ + λ(1−β₀(1)))`, it derives a closed form for
the first idle state from the Layer-1 boundary balance:

$$
P_0 = \frac{\lambda}{\mu + \gamma + \frac{\lambda}{2}\left(1-\beta_0(2)\right)}\, e^{-\lambda/\gamma},
\qquad
P_{0^{(n)}} = P_0 \prod_{k=2}^{n} \frac{\lambda}{k\gamma + \lambda(1-\beta_0(k))}.
$$

The key structural change: **μ now appears in the P₀ denominator.**

**Implementation check.** The β₀(n) recurrence reproduces the PDF's own worked
example exactly — for λ=1.5, μ=3.0, γ=1.0 it gives β₀(2) = 0.184470 (matching
p. 6) at truncation depths M = 6, 15, and 100, confirming the implementation is
faithful.

**Result: R² = −32.0** — *worse* than the −17.6 of the previous corrected variant.

Because μ ≈ 50 now sits in the P₀ denominator, P₀ shrinks by ≈ 14× (0.21 → 0.016),
so the spine sum Σ becomes *smaller*. But the validation shows the spine needs to
be *larger*, not smaller. The "needed Σ" below is back-solved from the
model-independent balance `P∧ + Σ = 1 − λ(1−loss)/μ`:

| λ | μ | γ | Σ (Response 2) | Σ needed | gap |
|---|---|---|---|---|---|
| 1.5 | 50 | 2.5  | 0.0198 | 0.4226 | 21× |
| 1.5 | 50 | 10.0 | 0.0229 | 0.1143 | 5× |
| 1.5 | 50 | 25.0 | 0.0193 | 0.0383 | 2× |

Response 2 pushes the spine sum *further* from its target, so loss rates become
more negative (e.g. −13.4 where the simulation gives 0.048).

**Why no spine anchor can fix this.** At utilisation ρ = λ/μ = 0.03 the system is
idle ≈ 97% of the time, so `P∧ + Σ_idle` must reach ≈ 0.97. The empty state alone
gives `P∧ = e^(−0.6) = 0.549`; the remaining ≈ 0.42 of idle mass lives in
**mixed-idle states** (`01`, `001`, `010`, … — head-bit 0, server idle). The
pure-zero spine `0^(n)` is a vanishing subset of the idle states and cannot carry
that mass *regardless of how P₀ is anchored*. Refining P₀ rearranges a term that is
already ≈ 20× too small.

---

## 3. What the data actually shows

The loss is governed by **competing exponentials at the queue head** — the head
packet is either *served* (rate μ) or *decays/expires* (rate γ):

$$
\boxed{\;\text{loss} \approx \dfrac{\gamma}{\mu+\gamma}, \qquad
\text{served} \approx \dfrac{\mu}{\mu+\gamma}\;}
$$

On the light-load subset (λ < μ, 3366 rows):

- `loss` vs `γ/(μ+γ)` : **R² = 0.984**
- `served = 1−loss` vs `μ/(μ+γ)` : **R² = 0.984**

Per-μ accuracy is even higher (the global R² is dragged down only by the heavily
overloaded μ = 50 rows):

| μ | n | R²(`γ/(μ+γ)`) | mean abs err |
|---|---|---|---|
| 50  | 4424 | 0.590 | 0.0898 |
| 100 | 879  | 0.986 | 0.0194 |
| **125** | 193 | **0.9994** | **0.0037** |
| 150 | 379  | 0.997 | 0.0099 |
| 175 | 474  | 0.986 | 0.0215 |
| 200 | 645  | 0.969 | 0.0331 |

**Connection to prior work.** `γ/(μ+γ)` is exactly the *γ-share* found in the
ratio analysis (`p₂ = S·γ/(μ+γ)`, `r = p₁/p₂ ≈ μ/γ`). The same μ-vs-γ competition
that sets the state-probability ratio also sets the loss fraction.

### Heavy-traffic correction

`γ/(μ+γ)` is the **light-load limit**. When λ > μ the queue saturates and overload
adds losses on top. The simple additive correction

$$
\text{loss} = \frac{\gamma}{\mu+\gamma} + \frac{\mu}{\mu+\gamma}\,\max\!\left(0,\; 1 - \frac{\mu}{\lambda}\right)
$$

raises the global fit to **R² = 0.972** and is exact in both limits (light load →
`γ/(μ+γ)`; extreme overload → 1). The residual structure (μ = 50 group at R² =
0.93) suggests a smoother λ-dependent crossover term remains to be identified —
a natural target for symbolic regression.

---

## 4. Conclusion

1. The **SBI continued-fraction formula** in `Matrix.pdf`, implemented exactly as
   written, **does not reproduce** the simulated loss rates (R² = −147, with
   unphysical negative values). Subtracting the empty-state mass P∧ from the
   throughput term (`λ − μ(1 − P∧ − Σ)`) improves this to R² = −17.6 but still
   fails; the **Response 2** closed-form spine anchor is worse still (R² = −32.0,
   it shrinks the spine sum in the wrong direction). The claimed `R² = 1.0` is not
   supported by the simulation data in `merged_all_infinite_buffer.csv`.

2. The failure is structural, not a normalisation artefact: the spine term
   `1 − P∧ − Σ P_{0^(n)}` is conflated with P(server busy), yet a direct check
   against the model-independent true P(busy) = λ(1−loss)/μ shows the two diverge
   by 6×–60× in opposite directions as γ varies. No rescaling of the loss rate
   can repair a γ-dependent error of this kind.

3. The simulated loss is well described by the **competing-exponential law
   `loss = γ/(μ+γ)`** (R² = 0.984 at light load, 0.9994 at μ = 125), consistent
   with the independently discovered ratio result `r ≈ μ/γ`. A one-term overload
   correction extends this to R² = 0.972 across all traffic regimes.

**Recommendation:** revisit the SBI derivation — specifically the identification
of the throughput term and the `R_n` spine multiplier (its denominator
`nγ + λ(1−β₀(n))` does not match the flux-balance relation
`R_{n+1} = (1−β₀(n))·λ/(nγ)` implied by Phase II). The competing-exponential law
provides a simpler, empirically validated alternative.
