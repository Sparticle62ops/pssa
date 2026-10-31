# Model math

The equations behind PSSA in one place, with the source references and the
measured results. The same reference renders in the dashboard's `F11` math tab
(see [assets/math.md](assets/math.md)); the architecture walk-through is
[docs/architecture.md](docs/architecture.md).

## Input and state update

```
x = γ ⊙ (e / √(mean(e²) + 10⁻⁵)) + β
Δ = softplus(WΔ x); B = WB x; C = WC x
Aᵢⱼ = −softplus(aᵢⱼ)
hᵢⱼ ← exp(Δᵢ Aᵢⱼ) hᵢⱼ + Δᵢ Bⱼ xᵢ
yᵢ = Σⱼ hᵢⱼ Cⱼ
```

Each latent channel carries a state vector across tokens. Positive Δ controls
its decay and input write; this is not an attention matrix.
Source: `src/pssa.rs:691-698,844-896`; `src/gpu_batch.rs:490-526`.

## Memory slots and retrieval

```
q = Wqx x + Wqh y; r = ||q||₂
p = q · min(r/(1+r), Rmax)/r   (p = 0 when r = 0)
d(p,k) = 2 asinh √(||p−k||² / ((1−||p||²)(1−||k||²)))
wⱼ = exp((dmin−dⱼ)/τ) / Σᵢ exp((dmin−dᵢ)/τ)
m = Σⱼ wⱼ vⱼ; g = σ(Wgate x)
```

Keys live inside the open Poincaré ball; `Rmax` is the representable safe
radius. Empty memory returns zero. Each depth block owns its own finite bank.
Source: `src/memory.rs:63-77,110-126,184-237`; `src/pssa.rs:893-914`.

## Plastic adapter and MLP

```
a = (Ufast + Uconsolidated) SiLU(D x)
z = y/√state + g ⊙ (Wproj m) + a
F(x) = z + W₂ SiLU(W₁ z); SiLU(u) = u σ(u)
```

Consolidation folds fast plastic updates back into the slow copy without
changing their effective sum in exact arithmetic:

```
Uconsolidated ← Uconsolidated + α Ufast;  Ufast ← (1−α) Ufast
```

The trainable adapter has rank 16; the MLP expands latent width by 2.
Source: `src/pssa.rs:556,916-945`; `src/adapter.rs:36-48,65-73`.

## Memory write and refractory gate

If chunk loss > 3.5: propose key `p` and value `F(x)` at the last token, final
loop. Unused slot: append. Full bank: consider only the circular write head.

```
ρ = max((1−exp(−Δstep/60))², 0.005)
damage = 0.08 · surprise · ρ
```

If confidence > damage: subtract damage, keep the slot; otherwise overwrite and
reset confidence to 1. The bank is a separate runtime memory, not trainable
model weights; the write path is a defense mechanism, not a gradient-based
delta-rule write. Source: `src/pssa.rs:2182-2204`; `src/memory.rs:129-182`;
`src/defense.rs:18-21,45-69`.

## Depth and shared loops

One loop: `h₀ = F₀(embed[token]); hₗ = hₗ₋₁ + Fₗ(hₗ₋₁)/√depth`. With multiple
loops, each block reuses its own weights but keeps a separate recurrent carry
per pass, so loops do not multiply parameter count and are runtime-only.
Source: `src/pssa.rs:702-768,1725-1780`.

## Readout and loss

```
logits = Wunembed h / √latent
lossₜ = max(logits) − logits[targetₜ] + ln Σᵢ exp(logitsᵢ − max(logits))
```

Source: `src/pssa.rs:1780-1785,1964-1994`.

## What the math produced

Two parameter-matched models, same corpus, tokenizer, optimizer schedule and
seed, 12.7M tokens of cleaned WikiText-103. Full methodology and caveats in
[docs/benchmarks.md](docs/benchmarks.md).

| Held-out slice, 198,939 unseen tokens | PSSA | Transformer |
| --- | --- | --- |
| Cross-entropy | **3.997** | 4.429 |
| Perplexity | **54.4** | 83.8 |
| Next-token accuracy | **24.1%** | 18.0% |

| Same CPU, fixed work | PSSA | Transformer |
| --- | --- | --- |
| Training throughput | **1,716 tok/s** | 415 tok/s (4.13x) |
| Generate 200 tokens | **226 ms** | 2,735 ms (12x faster) |

Every gradient is checked against the scalar CPU reference path: max
difference 2.98e-8 against a 3e-8 gate.
