# Input and state update
x = γ ⊙ (e / √(mean(e²) + 10⁻⁵)) + β
Δ = softplus(WΔ x); B = WB x; C = WC x
Aᵢⱼ = −softplus(aᵢⱼ)
hᵢⱼ ← exp(Δᵢ Aᵢⱼ) hᵢⱼ + Δᵢ Bⱼ xᵢ
yᵢ = Σⱼ hᵢⱼ Cⱼ
Each latent channel carries a state vector across tokens. Positive Δ controls its decay and input write; this is not an attention matrix.
Source: src/pssa.rs:691-698,844-896; src/gpu_batch.rs:490-526

# Memory slots and retrieval
q = Wqx x + Wqh y; r = ||q||₂
p = q · min(r/(1+r), Rmax)/r (p = 0 when r = 0)
d(p,k) = 2 asinh √(||p−k||² / ((1−||p||²)(1−||k||²)))
wⱼ = exp((dmin−dⱼ)/τ) / Σᵢ exp((dmin−dᵢ)/τ)
m = Σⱼ wⱼ vⱼ; g = σ(Wgate x)
Keys live inside the open Poincaré ball (Rmax is the representable safe radius). Empty memory returns zero. Each depth block owns its own finite bank.
Source: src/memory.rs:63-77,110-126,184-237; src/pssa.rs:893-914

# Plastic adapter and MLP
a = (Ufast + Uconsolidated) SiLU(D x)
z = y/√state + g ⊙ (Wproj m) + a
F(x) = z + W₂ SiLU(W₁ z); SiLU(u) = u σ(u)
Consolidation: Uconsolidated ← Uconsolidated + α Ufast; Ufast ← (1−α) Ufast.
The trainable adapter has rank 16; the MLP expands latent width by 2. Consolidation transfers coefficients without changing their effective sum in exact arithmetic; the slow copy is not an independent trainable matrix.
Source: src/pssa.rs:556,916-945; src/adapter.rs:36-48,65-73

# Memory write and refractory gate
If chunk loss > 3.5: propose key p and value F(x) at the last token, final loop.
Unused slot: append. Full bank: consider only the circular write head.
ρ = max((1−exp(−Δstep/60))², 0.005)
damage = 0.08 · surprise · ρ
If confidence > damage: subtract damage, keep slot; else overwrite and reset confidence to 1.
The bank is a separate runtime memory, not trainable model weights. This defense path is not a gradient-based delta-rule write. Δstep is optimizer steps at this call site.
Source: src/pssa.rs:2182-2204; src/memory.rs:129-182; src/defense.rs:18-21,45-69

# Depth and shared loops
One loop: h₀ = F₀(embed[token]); hₗ = hₗ₋₁ + Fₗ(hₗ₋₁)/√depth for l > 0.
Multiple loops: Rₗ(u) starts at u; repeat u ← u + Fₗ,pass(u)/loops.
h₀ = R₀(embed[token]); hₗ = hₗ₋₁ + Rₗ(hₗ₋₁)/√depth for l > 0.
Each depth block owns independent parameters. Loops reuse that block's weights but keep separate recurrent carry per pass. Loops do not multiply parameter count and are runtime-only (not saved in checkpoints).
Source: src/pssa.rs:702-768,1725-1780

# Readout and loss
logits = Wunembed h / √latent
lossₜ = max(logits) − logits[targetₜ] + ln Σᵢ exp(logitsᵢ − max(logits))
chunk loss = meanₜ(lossₜ)
The heatmap reports the raw softmax probability before decoding controls. Low probability means uncertainty, not a training fault. Matrix MAC estimates exclude nonlinearities, memory distance/retrieval, normalization, and backward/optimizer work.
Source: src/pssa.rs:1780-1785,1964-1994
