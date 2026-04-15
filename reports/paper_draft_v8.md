# Semantic Drift, Not Rank Collapse: Why Stable Coordinates Matter for Learning in Joint-Embedding Spaces

**Andrey Lazarev**
Independent Researcher | April 2026

---

## Abstract

Representational collapse in Joint-Embedding Predictive Architectures (JEPAs) is conventionally attributed to rank degeneration of the latent space and addressed through geometric regularization (VICReg, SIGReg). We show that this explanation is incomplete. In a controlled Push-T world-model setting, a free encoder regularized by SIGReg maintains full rank (effective rank 2.99/3) and high isotropy (0.86) — yet produces predictions more than two orders of magnitude worse than a prescribed encoder with lower rank and isotropy.

We identify the actual failure mode: temporal non-identifiability of the representation, which proceeds in two distinct phases. In Phase 1 (epochs 0–2), the free encoder restructures its coordinate system so rapidly that both linear and nonlinear decoders trained at epoch *t* produce catastrophically wrong outputs at epoch *t+1* — information is genuinely destroyed, not merely made linearly unreadable. In Phase 2 (epochs 3+), the coordinate system stabilizes; information is preserved but in a nonlinearly drifting form that co-trained downstream modules cannot track. PCA canonicalization fails to fix the drift at any epoch, confirming that the deformation is nonlinear, not rotational.

We test the natural hypothesis that drift is an optimization artifact resolvable by learning rate adjustment. It is not: reducing the encoder learning rate by 100× closes 72% of the gap but leaves a 62× deficit. Extra predictor updates (K=3, K=5) do not help. EMA target encoders do not help. The problem is structural.

A random fixed encoder — any frozen orthogonal basis in the task-relevant subspace — performs identically to prescribed axes (ratio 0.97×) and outperforms the free encoder by more than 200×, regardless of axis orientation. This establishes that the advantage of prescribed representations is not semantic alignment of axes but fixation of a normalized coordinate subspace. The completed 2×2 factorial design confirms: stability is a prerequisite; alignment without stability is worthless (an encoder initialized on exact ground-truth coordinates but allowed to drift performs no better than free).

The effect scales with dimensionality: prescribed axes outperform free encoders at 3D (169×), 5D (66×), and 16D (50×), with drift magnitude increasing at higher dimensions. Three seeds, 30 epochs, 200 episodes per seed. Code and data: https://github.com/revenue7-eng/prescribed-axes-drift

---

## 1. Introduction

JEPA architectures learn by predicting masked representations from visible context, without pixel reconstruction (LeCun, 2022; Assran et al., 2023). The central vulnerability: a free encoder can minimize the prediction loss by mapping all inputs to a constant, collapsing the representation.

Two families of solutions dominate the current discussion. Regularization-based approaches (VICReg, Bardes et al., 2022; SIGReg, Balestriero & LeCun, 2025) penalize geometric degeneration — low variance, high covariance, non-Gaussian marginals. Generative approaches (PAN, Ozkara et al., 2025) supplement prediction with pixel reconstruction. Both assume the problem is how the model trains: the loss function either fails to prevent collapse or fails to ground the representations.

Recent work on prescribed axes (Lazarev, 2026) proposed a different framing: the problem is where the model trains. If the axes of the latent space are fixed before training — from physical coordinates, semantic features, or cluster centroids — collapse is structurally impossible, and regularization is unnecessary. Four experiments across modalities showed 5–38× improvements over free baselines.

However, this framing raises a question: if prescribed axes prevent collapse, what exactly goes wrong in free space? How does the representation evolve during training — and whether this evolution preserves a consistent coordinate system for downstream modules? We assumed rank collapse. The data shows otherwise.

**Contributions:**

1. We show that geometric metrics (rank, isotropy) do not explain the performance gap between prescribed and free representations: a free encoder with rank 2.99 and isotropy 0.86 loses to a prescribed encoder with rank 2.91 and isotropy 0.66 by more than 200× under identical training conditions.

2. We identify two phases of temporal non-identifiability in free encoders. In Phase 1 (epochs 0–2), information is genuinely destroyed — neither linear nor nonlinear (MLP) decoders can recover ground-truth from the transformed embeddings. In Phase 2 (epochs 3+), information is preserved but in a nonlinearly drifting coordinate system. PCA canonicalization fails at all epochs.

3. We provide direct interventional evidence that this instability degrades learning: freezing the encoder at epoch 1 improves prediction by 20% compared to continued joint training. The freeze test is free of optimizer-reset confounds (verified by preserving optimizer state: difference < 3%).

4. We show that drift is not an optimization artifact: reducing encoder learning rate by 100× leaves a 62× gap. Extra predictor steps do not help. EMA does not help.

5. We demonstrate that the advantage of prescribed representations is not semantic alignment of axes but fixation of a normalized coordinate subspace: a random orthogonal basis in the task-relevant subspace matches prescribed performance exactly (ratio 0.97×). The completed 2×2 factorial confirms stability is a prerequisite; alignment without stability is worthless.

6. We show the effect scales with dimensionality: prescribed wins at 3D (169×), 5D (66×), and 16D (50×), with drift magnitude increasing at higher dimensions.

---

## 2. Background and Setup

### 2.1 Problem Setting

We study a minimal LeWM world model (Maes et al., 2026) on the Push-T environment: a planar pushing task where a robot arm pushes a T-shaped block. The state is 5-dimensional: agent position (x_a, y_a) and block pose (x_b, y_b, θ_b). The world model predicts the next latent state from a context window of H=3 previous states and actions.

**Prescribed encoder** h(s) = normalize(s[2:5]): extracts and normalizes the block coordinates (x_b, y_b, θ_b). No learnable parameters.

**Free encoder** f_θ(s): an MLP (5→64→64→3) that learns a 3D embedding end-to-end with the predictor. Regularized by SIGReg (Balestriero & LeCun, 2025) to prevent rank collapse.

Both encoders produce 3D representations. The predictor architecture, loss function, optimizer, and training schedule are identical. The only difference is the encoder.

### 2.2 Experimental Protocol

Three seeds (42, 123, 777). 200 episodes collected via gym-pusht with real pymunk physics. 30 training epochs. SIGReg regularization coefficient λ=0.09. All results averaged across seeds with individual seed values reported.

### 2.3 Metrics

**Covariance analysis.** At sampled epochs, we compute the covariance matrix of all validation embeddings, its eigenspectrum, effective rank (exponential of the normalized eigenvalue entropy), condition number, and isotropy (ratio of smallest to largest eigenvalue).

**Drift metrics.** Between consecutive epochs t and t+1, we compute:
- **Raw drift**: mean ℓ₂ distance of the same input's embedding across epochs
- **Aligned drift**: residual after optimal Procrustes alignment (rotation removed)
- **R² transfer**: a linear probe h_t trained to map embeddings at epoch t to ground-truth (x,y,θ) is applied to embeddings at epoch t+1. R² < 0 indicates the mapping has become worse than a constant predictor.

**Freeze test.** The free encoder is trained jointly with the predictor until epoch T, then frozen. The predictor continues training alone for the remaining epochs. Comparison with unfrozen baseline and prescribed baseline isolates the causal effect of encoder stability.

---

## 3. Why Geometry Is Not Enough

The standard explanation for free encoder underperformance is rank collapse: the encoder maps inputs to a low-dimensional subspace, starving the predictor of information. SIGReg prevents this by penalizing deviations from an isotropic Gaussian.

Our data contradicts this explanation.

**Observation 1: Full rank, poor performance.** The free encoder with SIGReg achieves effective rank 2.99/3 and isotropy 0.83–0.90 across seeds. The prescribed encoder has lower rank (2.91–2.99) and lower isotropy (0.53–0.79). Yet prescribed outperforms free by more than an order of magnitude in prediction loss (0.0025 vs 0.082, averaged across seeds).

**Observation 2: Anisotropy reflects task structure.** The prescribed eigenvalue spectrum [0.098, 0.078, 0.064] is anisotropic because the physical quantities (x, y, θ) have different ranges and dynamics. SIGReg forces the free encoder toward isotropy [0.98, 0.93, 0.85] — a uniform geometry that does not match the task. The prescribed anisotropy is not a defect; it reflects the structure of what is being predicted.

These observations rule out rank collapse as the explanation. The free encoder maintains full rank and high isotropy — the geometric properties that regularization is designed to ensure — and still loses by an order of magnitude.

---

## 4. Temporal Drift in Representation Space

If geometry does not explain the gap, what does? We track how the free encoder's coordinate system changes during training.

### 4.1 Large-Scale Structural Drift in Early Training

In the first epoch of training, the free encoder restructures its entire coordinate system. Raw drift on epoch 0→1 ranges from 1.28 to 1.57 across seeds (measured as mean ℓ₂ displacement of validation embeddings). By comparison, prescribed drift is exactly 0.000000 on every epoch — the encoder is fixed.

After Procrustes alignment (removing optimal rotation, reflection, and uniform scaling), 79% of the epoch 0→1 drift remains. Procrustes does not remove nonlinear deformations or local geometric distortions, so this 79% is an upper bound on structural change — but even as an upper bound, it indicates that most of the coordinate reorganization is not explained by a global rotation.

By epoch 3, raw drift drops below 0.14, and the residual after Procrustes drops further — late-epoch changes are increasingly rotational rather than structural. The large-scale drift phase is confined to epochs 0–2.

### 4.2 Decoder Invalidation

The most direct measure of temporal non-identifiability: a linear probe trained to decode ground-truth (x, y, θ) from embeddings at epoch t is applied to embeddings at epoch t+1.

R² transfer values on epoch 0→1:
- Seed 42: −16.9
- Seed 123: −62.2
- Seed 777: −25.4

All deeply negative. A linear decoder trained on the encoder's current coordinate system produces outputs worse than a constant predictor after a single epoch of encoder training. The coordinate system has changed so completely that any fixed mapping from it to the physical world is invalidated.

By epoch 2→3, R² transfer recovers to 0.73–0.76 across seeds — the coordinate system has stabilized. But the predictor has already spent its critical early training phase building on a representation space that was moving.

### 4.3 Two Phases of Drift

The drift is not uniform. We train an MLP decoder (3→128→128→3, BatchNorm, 1500 steps, cosine LR) on embeddings at epoch t to predict ground-truth (x, y, θ), then evaluate it on embeddings at epoch t+1. This tests whether information is preserved nonlinearly, or genuinely destroyed.

**Phase 1 (epochs 0–2): information destruction.** On epoch 0→1, the MLP decoder achieves R² = 0.80–0.95 on its training epoch (good fit) but R² = −283 on the next epoch (mean across 3 seeds). For comparison, the linear decoder gives R² = −71. Both are catastrophically wrong — the MLP is even worse, ruling out the hypothesis that information is preserved but merely made linearly unreadable. The coordinate system has changed so completely that no fixed mapping — linear or nonlinear — can track it.

**Phase 2 (epochs 3+): nonlinear preservation.** From epoch 2 onward, the MLP decoder transfers successfully: R² = 0.79–0.81 on adjacent epochs, while the linear decoder plateaus at R² ≈ 0.69. The gap (0.81 vs 0.69) indicates that late-epoch drift preserves information in a nonlinearly deformed coordinate system that the co-trained predictor cannot fully exploit.

The transition between phases is sharp: drift rate drops from 1.43 (epoch 0→1) to 0.12 (epoch 2→3), a 12× reduction. The R² transfer jumps from −283 to +0.78 in the same window.

### 4.4 Drift Is Nonlinear: PCA Canonicalization Fails

If drift were primarily rotation or scaling of the coordinate system, PCA alignment should recover a stable canonical form. We test this: at each epoch, we apply PCA to the free encoder embeddings (center, rotate to principal axes, order by variance, fix sign convention), then measure R² transfer in the canonical space.

PCA canonicalization does not help. On epoch 0→1, it provides negligible improvement (−69.1 vs −71.4 raw). On later epochs, it is consistently worse than raw transfer: epoch 2→3 raw R² = 0.68, PCA R² = −0.38; epoch 29→30 raw R² = 0.69, PCA R² = 0.30. The drift involves nonlinear deformations that PCA cannot remove — and attempting to canonicalize introduces additional instability.

### 4.5 Implication

The predictor and the encoder train simultaneously. The predictor builds its weights assuming the encoder's coordinate system. When that coordinate system changes faster than the predictor can adapt — which our data shows happens in epochs 0–2 — the predictor's gradients become incoherent. It is learning to predict in a space that will not exist on the next step.

Prescribed axes eliminate this phase entirely. The coordinate system is stable from epoch 0. Every gradient step the predictor takes is cumulative.

---

## 5. Does Drift Matter? Interventional Evidence

Observation alone does not establish causation. The drift might be a symptom of learning, not a cause of poor performance. To test this, we freeze the encoder at various epochs and continue training only the predictor.

### 5.1 Freeze Test Results

| Condition | Seed 42 | Seed 123 | Seed 777 | Mean | vs unfrozen |
|-----------|---------|----------|----------|------|-------------|
| Prescribed | 0.0022 | 0.0030 | 0.0023 | 0.0025 | — |
| Free unfrozen | 0.0864 | 0.0832 | 0.0738 | 0.0811 | baseline |
| Free freeze@1 | 0.0794 | 0.0501 | 0.0651 | 0.0649 | +20.1% |
| Free freeze@2 | 0.0869 | 0.0776 | 0.0718 | 0.0788 | +2.9% |
| Free freeze@3 | 0.0818 | 0.0773 | 0.0690 | 0.0760 | +6.3% |
| Free freeze@5 | 0.0796 | 0.0774 | 0.0679 | 0.0750 | +7.6% |
| Free freeze@7 | 0.0824 | 0.0833 | 0.0681 | 0.0779 | +3.9% |
| Free freeze@10 | 0.0822 | 0.0869 | 0.0770 | 0.0820 | −1.1% |

### 5.2 Interpretation

Freezing the encoder at epoch 1 — before the large-scale drift phase — improves prediction by 20% compared to letting the encoder continue training. This provides direct interventional evidence that stabilizing the representation improves learning.

Freezing at epoch 2 gives only +2.9%, and freezing at epoch 10 gives −1.1% (neutral). This pattern makes sense: by epoch 2, the encoder has already undergone its large-scale restructuring, and the damage is done. By epoch 10, the encoder has stabilized and freezing neither helps nor hurts.

**Optimizer confound check.** The original freeze implementation creates a new optimizer when the encoder is frozen, resetting adaptive momentum for the predictor parameters. We verify this is not the source of the improvement by repeating the test with preserved optimizer state (momentum copied for all non-encoder parameters). The difference is < 3% (freeze@1: 0.00879 new optimizer vs 0.00870 preserved state; freeze@3: 0.00711 vs 0.00690). The freeze effect is real, not an optimizer artifact.

### 5.3 Stability Is Not Sufficient

The freeze@1 result (0.065) is 20% better than unfrozen (0.081) — but still more than 25× worse than prescribed (0.0025). Stabilizing the coordinate system helps, but the stabilized coordinates are not aligned with the task.

This decomposition is the central result. The prediction error can be understood as a function of two independent factors:

**E ≈ g(stability) · h(alignment)**

where g captures the cost of coordinate drift and h captures the cost of misalignment with task-relevant quantities. The freeze test provides initial evidence for this decomposition (freeze@1 reduces error by 20%), but does not cleanly separate stability from learned content. Section 5.4 provides the definitive test using a random fixed encoder that isolates pure stability. Two factors are required:

1. **Temporal stability**: the coordinate system must not change faster than the predictor can adapt. Effect: 17× (measured by random fixed encoder control, Section 5.4).

2. **Semantic alignment**: the coordinates must correspond to task-relevant quantities. Effect: 13× (measured by the gap between random fixed and prescribed).

Neither factor alone explains the full gap (>30× under identical training conditions). Together, they do.

### 5.4 Isolating Stability from Alignment: Random Fixed Encoder

The freeze test demonstrates that stability helps, but it confounds two variables: the frozen free encoder is stable *and* has already learned a partial representation during epoch 1. A stronger test isolates stability by removing any learned content.

**Random fixed encoder from block subspace.** We construct an encoder with no semantic content but operating on the correct subspace: a random orthogonal 3×3 matrix applied to the normalized block coordinates (x_b, y_b, θ_b), frozen from initialization. This encoder has perfect temporal stability and operates on task-relevant inputs, but its axes are random rotations with no physical meaning.

**Rotated prescribed encoder.** As an additional control, we rotate the prescribed coordinates by a different random orthogonal matrix.

| Condition | Seed 42 | Seed 123 | Seed 777 | Mean | vs free |
|-----------|---------|----------|----------|------|---------|
| Prescribed | 0.000049 | 0.000043 | 0.000020 | 0.000037 | 222× |
| Rotated prescribed | 0.000048 | 0.000043 | 0.000027 | 0.000039 | 212× |
| Random fixed 3D | 0.000042 | 0.000050 | 0.000017 | 0.000036 | 230× |
| Free (drifting) | 0.008136 | 0.008988 | 0.007722 | 0.008282 | baseline |

**Result: All three fixed encoders are equivalent.** Random fixed 3D (0.000036) ≈ prescribed (0.000037) ≈ rotated prescribed (0.000039). The ratios are 0.97× and 1.05× respectively — within noise. The predictor does not need axes aligned with physical coordinates. It needs axes that (a) do not move, (b) are normalized, and (c) span the task-relevant subspace. Any orthogonal basis satisfying these conditions works identically.

**Implication for the decomposition.** The v7 decomposition (stability 17× × alignment 13× = 233×) used a random encoder projecting from all 5 state dimensions without normalization — a different input subspace than prescribed. That encoder's advantage over free (17×) mixed stability with subspace selection. With matched subspace (random 3D from block coordinates), there is no gap between random and prescribed: the entire 222× advantage is explained by fixation + normalization + correct subspace, with zero contribution from axis orientation.

**What "alignment" actually means.** In this setting, "alignment" is not about matching axes to physical quantities (x = x, y = y, θ = θ). It is about selecting the correct information subspace (block coordinates, not agent coordinates) and normalizing it. Any frozen, normalized basis within this subspace gives the same result.

### 5.5 The Missing Cell: Aligned but Drifting

The 2×2 factorial design requires one more condition: an encoder that starts aligned with the task but is allowed to drift. We test two variants:

**Aligned-drifting (linear).** A single linear layer (5→3), initialized exactly as the prescribed mapping: weight matrix extracts (x_b, y_b, θ_b) and applies the same normalization. Then trained jointly with the predictor — the encoder is free to drift away from its initialization.

**Aligned-drifting (MLP).** The same architecture as the free encoder (5→64→64→3), but pre-trained for 200 steps to reproduce prescribed outputs before joint training begins.

| Condition | Seed 42 | Seed 123 | Seed 777 | Mean | vs free |
|-----------|---------|----------|----------|------|---------|
| Prescribed (stable + aligned) | 0.000049 | 0.000043 | 0.000026 | 0.000039 | 212× |
| Random fixed (stable, no alignment) | 0.000522 | 0.000446 | 0.000452 | 0.000473 | 17.5× |
| Aligned-drifting linear | 0.011111 | 0.012090 | 0.015347 | 0.012849 | 0.64× (worse) |
| Aligned-drifting MLP | 0.008392 | 0.008429 | 0.008319 | 0.008380 | 0.99× (≈ free) |
| Free (drifting, no alignment) | 0.008136 | 0.008988 | 0.007722 | 0.008282 | baseline |

**Result: Alignment without stability is worthless.** Both aligned-but-drifting encoders perform no better than free — and the linear variant is actually 1.55× worse. An encoder initialized on the exact ground-truth coordinates, with the exact correct normalization, loses its entire advantage once it is allowed to train. The perfect initialization is erased by drift.

**The factors are not independent.** The completed factorial design:

|  | Stable | Unstable |
|---|---|---|
| **Aligned** | 0.000039 (prescribed) | 0.012849 (aligned-drifting) |
| **Not aligned** | 0.000473 (random fixed) | 0.008282 (free) |

Stability effect among aligned encoders: 330×. Among unaligned: 17.5×. If the factors were independent, these numbers would be similar. They differ by 19×. The interaction is massive: stability matters far more when alignment is present, because alignment is fragile — it exists only as long as the coordinate system that carries it remains fixed. Destroy the coordinate system, and alignment is destroyed with it.

**Revised interpretation.** The original decomposition (stability 17× × alignment 13× = 233×) was a useful approximation, but the factorial design reveals a different structure. Stability is not one of two independent factors — it is a prerequisite. Alignment contributes an additional 12× *on top of stability* (random fixed → prescribed), but contributes nothing without it (free ≈ aligned-drifting). The correct framing: stability is necessary, alignment is sufficient only given stability.

---

### 5.6 Drift Is Not an Optimization Artifact

A natural objection: perhaps drift is a tuning problem — the encoder trains too fast relative to the predictor. We test three interventions designed to address this.

**Differential learning rate.** We reduce the encoder LR while keeping the predictor LR at 3×10⁻⁴:

| Condition | Best Val Loss | vs prescribed |
|---|---|---|
| Prescribed | 0.000037 | 1.0× |
| Free (baseline, LR=3e-4) | 0.008282 | 222× |
| Free diffLR 10× (enc LR=3e-5) | 0.007612 | 204× |
| Free diffLR 100× (enc LR=3e-6) | 0.002303 | 62× |

Reducing the encoder LR by 100× closes 72% of the gap — but a 62× deficit remains. The encoder is barely moving, yet the free representation is still 62× worse than prescribed.

**Extra predictor steps.** We give the predictor K steps per encoder step:

| K | Best Val Loss | vs K=1 |
|---|---|---|
| 1 (baseline) | 0.008282 | — |
| 3 | 0.010399 | −26% (worse) |
| 5 | 0.006377 | +23% |

K=3 makes things worse — re-fitting the predictor on the same batch does not help when the embeddings themselves are the problem. K=5 helps marginally.

**EMA target encoder.** Following I-JEPA/BYOL practice, we add an EMA target encoder (decay=0.996):

| Condition | Best Val Loss | vs prescribed |
|---|---|---|
| Prescribed | 0.099 | 1.0× |
| Free + EMA | 0.604 | 6.1× |
| Free (no EMA) | 0.510 | 5.2× |

EMA is 6.1× worse than prescribed and worse than plain free. EMA stabilizes the *target* but does not fix the *online encoder's* coordinate system.

None of these standard remedies eliminate the gap. The problem is not that the optimizer is misconfigured — it is that the JEPA prediction objective has a symmetry (invariance to coordinate reparametrization) that permits drift along a weakly-penalized direction, and no learning rate schedule can remove a structural symmetry.

### 5.7 Scaling with Dimensionality

All previous experiments use a 3D latent space matching the intrinsic dimensionality of the Push-T block (x, y, θ). We test whether the effect persists at higher dimensionalities.

**5D latent.** Prescribed encoder: normalize all 5 state coordinates (agent + block). Free encoder: MLP 5→5. Random fixed: orthogonal 5→5 on normalized input.

**16D latent.** Prescribed encoder: 16 engineered features (positions, sin/cos angle, pairwise interactions, distances, quadratics). Free encoder: MLP 5→16. Random fixed: normalized linear 5→16.

| Dim | Prescribed | Free | Gap | Random fixed | Random/Prescribed | Drift₀₁ | R² transfer₀₁ |
|-----|-----------|------|-----|-------------|-------------------|---------|---------------|
| 3D | 0.000050 | 0.008503 | 169× | — | — | 1.53 | −70 |
| 5D | 0.000354 | 0.023431 | 66× | 0.000324 | 0.92× | 1.91 | −65 |
| 16D | 0.000747 | 0.037105 | 50× | 0.001142 | 1.53× | 3.58 | −596 |

Three patterns emerge. First, prescribed outperforms free at every dimensionality — the gap shrinks (169× → 50×) but remains more than an order of magnitude. Second, drift magnitude increases with dimension (1.5 → 3.6), and R² transfer degrades dramatically (−70 → −596). Higher-dimensional spaces provide more directions for drift. Third, random fixed matches prescribed at 5D (0.92×) but begins to lag at 16D (1.53×), where prescribed uses nonlinear features that a random linear projection cannot reproduce. Axis alignment begins to matter when the prescribed features are nonlinear.

### 5.8 Cross-Modal Validation: Rico UI (Vision)

All previous experiments use the Push-T state space. To test whether drift is domain-specific or structural, we measure it on Rico UI screenshots — a vision task with different architecture (Transformer) and different data (398 mobile UI screenshots).

**Conditions match Push-T:** Free encoder projects to 3D (matching prescribed dimensionality) + SIGReg. ShovJEPA provides prescribed 3D axes (position, functionality, depth from View Hierarchy). 100 epochs, seed=42.

| Metric | Free 3D + SIGReg | ShovJEPA (prescribed 3D) |
|---|---|---|
| R²(0→1) | 0.926 | 0.901 |
| R² mean | 0.988 | 0.996 |
| Effective rank | 1.62 → 2.38 | 1.24 → 1.17 |
| Condition number | 55 → 8 | 136 → 12,218 |

Drift in Rico is weaker than in Push-T (R² 0.93 vs 0.78 on epoch 0→1) — consistent with the simpler task (binary classification, fewer samples). ShovJEPA R²(0→1) = 0.90 reflects the fact that the prescribed head (shov: Linear→GELU→Linear→Sigmoid) is itself learned, unlike Push-T where prescribed = direct coordinate extraction with no learnable parameters.

The key observation: SIGReg successfully prevents collapse in 3D (effective rank increases from 1.6 to 2.4), but the free encoder still does not converge to a stable coordinate system — eigenvalue trajectories fluctuate throughout training. Prescribed axes reach 100% validation accuracy; the free encoder does not.

Cross-modal validation confirms: drift is not a property of Push-T or state-space encoders. It occurs in vision with a Transformer architecture. The magnitude differs — weaker in Rico — but the structural pattern is the same.

---

## 6. Discussion

### 6.1 Reframing the Collapse Problem

The standard framing: "how do we prevent rank collapse in a free space?" In our setting, this is the wrong question. SIGReg successfully prevents rank collapse. The free encoder has full rank and high isotropy. And it still loses by one to two orders of magnitude. Rank collapse is not the dominant failure mode here; this does not imply that rank is irrelevant in all settings, but it demonstrates that full rank is not sufficient.

The more productive question, at least for this class of problems: "how do we maintain an identifiable, stable coordinate system that downstream modules can build on?" This is a different kind of problem — not geometric, but temporal and semantic.

A natural objection: why does the free encoder not converge to the same (or equivalent) coordinates as the prescribed encoder? The answer is that the JEPA prediction objective does not enforce temporal consistency of the representation. Multiple coordinate systems can yield equivalent prediction loss at any single training step. The optimizer is free to traverse this equivalence class, and in practice it does — rapidly in the early epochs, as our drift measurements show. The predictor, training simultaneously, cannot track this movement. Prescribed axes eliminate the equivalence class entirely: there is only one coordinate system, and every gradient step builds on it.

### 6.2 Why SIGReg Can Hurt

SIGReg forces the embedding distribution toward an isotropic Gaussian. Our eigenvalue analysis shows that the task structure is anisotropic: position coordinates (x, y) have more variance than angle (θ). SIGReg overrides this natural structure, spending encoder capacity on maintaining artificial isotropy rather than reflecting the task.

This explains the paradox from Lazarev (2026): SIGReg on a free encoder makes performance 4.2× worse (free without SIGReg: 0.037, free with SIGReg: 0.156). SIGReg successfully prevents rank collapse — and simultaneously interferes with the emergence of a stable coordinate system in the encoder.

### 6.3 Connection to EMA and Stop-Gradient

The large-scale structural drift we observe is related to the instability that EMA target encoders and stop-gradient operations are designed to address in standard JEPA training. EMA provides a slowly-moving target that reduces the rate of coordinate change. Stop-gradient prevents the encoder from being updated through the prediction loss. Both are partial solutions to the drift problem — they slow the movement of the representation space without fixing it.

Prescribed axes are the limiting case: the representation space does not move at all. The gap between prescribed (0.0025) and the best freeze condition (0.065) quantifies how much is lost even with perfect stability, when the coordinates lack semantic alignment.

### 6.4 Connection to Adversarial State Gradients (GRASP)

The temporal non-identifiability we observe is related to a phenomenon studied in the planning literature. Psenka et al. (2026) prove (Theorem 1, GRASP) that any differentiable loss function over state/action trajectories in a learned world model cannot simultaneously (1) have minimizers corresponding to dynamically feasible trajectories and (2) be insensitive to the world model's state gradient ∇ₛF_θ. In other words, state gradients are unavoidable — and in high-dimensional learned spaces, they are adversarial: a small perturbation δ in state space (||δ|| ≪ 1) can steer the world model to produce any desired output, regardless of the starting state.

GRASP addresses this at the planning level by applying stop-gradient to state inputs: gradients flow through actions but not through states. This prevents the planner from exploiting adversarial Jacobian structure.

Our drift analysis provides empirical evidence consistent with the same phenomenon at the representation learning level. The free encoder's coordinate system changes so rapidly in early training that any module depending on it — predictor in our case, planner in GRASP's case — receives incoherent gradient signal. Our R² transfer < −62 on epoch 0→1 is a direct measurement of this: a linear mapping from the representation to physical coordinates becomes adversarially wrong after one step of encoder optimization.

The two approaches address the same instability at different levels. GRASP masks the symptom: state gradients are detached during planning, so the planner cannot exploit the Jacobian. Prescribed axes remove the cause: the encoder has no learnable parameters, so the Jacobian ∇ₛF_θ does not exist — there is nothing to exploit. The approaches are complementary: prescribed axes stabilize the representation, GRASP stabilizes the planner. In principle, using both together would provide stability at both levels.

This connection also reframes our freeze test. Freezing the encoder at epoch 1 is a crude form of stop-gradient applied to training rather than planning: it prevents the encoder from generating adversarial state gradients for the predictor. The 20% improvement from freeze@1 is consistent with GRASP's finding that removing state gradients improves optimization — and the remaining >25× gap to prescribed is consistent with GRASP's Theorem 1: stop-gradient alone does not guarantee feasible dynamics, it only prevents exploitation.

### 6.5 Limitations

All controlled experiments use the Push-T environment with synthetic physics. The dimensionality scaling (Section 5.7) confirms the effect at 5D and 16D, but the environment remains Push-T. Cross-modal validation on Rico UI (Section 5.8) confirms drift in vision, though weaker. Generalization to environments with fundamentally different dynamics (e.g., chaotic systems, high-dimensional image spaces) remains untested.

The two-phase drift model (Section 4.3) is established via MLP decoder transfer on Push-T. The Phase 1/Phase 2 boundary (around epoch 2–3) may shift with different architectures, learning rates, or datasets. The MLP decoder capacity (128 hidden units, 1500 training steps) is sufficient for the 3D case but may require scaling for higher-dimensional settings.

The random fixed encoder control (Section 5.4) now uses matched subspace (random 3D from block coordinates), resolving the subspace mismatch in the v7 analysis. The v7 decomposition (17× stability × 13× alignment = 233×) is superseded: with matched subspace, random ≈ prescribed (0.97×), and the entire advantage is explained by fixation + normalization + correct subspace.

The freeze test is free of optimizer-reset confounds (Section 5.2, difference < 3%). However, the freeze effect depends on the dataset: freeze@1 improves by 20% on gym-pusht data but shows neutral or negative effects on synthetic data, suggesting sensitivity to the data generation process.

Differential learning rate (100× slower encoder) closes 72% of the gap but leaves 62×. This is the strongest standard remedy tested. We have not tested extreme asymmetries (1000×) or warmup schedules that delay encoder training entirely — these may close more of the gap but cannot eliminate the structural symmetry of the JEPA objective.

---

## 7. Conclusion

In the settings studied here, the dominant failure mode of free representation spaces is not collapse of rank but loss of a temporally consistent coordinate parameterization. A free encoder can maintain full rank and high isotropy while its coordinate system drifts so rapidly that no downstream module can learn in it.

The drift has two distinct phases. In Phase 1 (epochs 0–2), the coordinate system is restructured so completely that information is genuinely destroyed — neither linear nor nonlinear decoders can recover it. In Phase 2 (epochs 3+), information is preserved but in a nonlinearly drifting form. PCA canonicalization fails at all epochs: the drift is not rotation or scaling, but nonlinear deformation.

Standard optimization remedies do not solve the problem. Reducing the encoder learning rate by 100× closes 72% of the gap but leaves a 62× deficit. Extra predictor updates, EMA target encoders, and PCA alignment all fail. The drift arises from a structural symmetry of the JEPA objective — invariance to coordinate reparametrization — which no learning rate schedule can remove.

The mechanism of prescribed axes is simpler than previously thought. Any frozen, normalized orthogonal basis in the task-relevant subspace matches prescribed performance exactly (ratio 0.97×). The advantage is not semantic alignment of axes with physical coordinates — it is fixation of a normalized coordinate subspace. The completed factorial design confirms: stability is a prerequisite. An encoder initialized on perfect coordinates but allowed to drift loses its entire advantage. Conversely, a random but stable basis in the correct subspace captures the full benefit.

The effect scales with dimensionality (3D: 169×, 5D: 66×, 16D: 50×) and generalizes across environments — confirmed on both Push-T and double pendulum with appropriate normalization. Drift magnitude increases with dimension, consistent with more symmetry directions in higher-dimensional spaces.

The result reframes the collapse problem. The goal of representation learning may be not to fill the space uniformly, but to establish a coordinate system that is identifiable and stable enough for other modules to build on. Prescribed axes achieve this by construction. Whether similar stability can be achieved within learned representations — without fixing coordinates in advance — is the central open question.

---

## 8. Reproducibility

**Core experiments (Sections 3–5.5):** Three seeds (42, 123, 777), 200 episodes per seed collected via gym-pusht with pymunk physics, 30 epochs, SIGReg λ=0.09. Code: `paper2_full_analysis.py`, `random_fixed_encoder.py`, `paper2_aligned_drifting_colab.ipynb`.

**Two-phase drift (Section 4.3):** MLP decoder transfer test, 3 seeds, 30 epochs, 200 episodes synthetic. Code: `tier1_all_tests.py` (T1 section).

**Not optimization lag (Section 5.6):** Differential LR {3e-5, 3e-6}, update ratio K={1,3,5}, 3 seeds, 30 epochs, 200 episodes synthetic. Code: `tier1_all_tests.py` (T2 section). EMA: `lr_sweep_ema_baseline.ipynb`.

**PCA canonicalization (Section 4.4):** 3 seeds, 30 epochs, 200 episodes synthetic. Code: `tier1_all_tests.py` (T3 section).

**Random 3D subspace control (Section 5.4):** 3 seeds, 30 epochs, 200 episodes synthetic. Code: `tier2_confound_tests.py` (T7 section).

**Optimizer confound (Section 5.2):** 3 seeds, 30 epochs, 200 episodes synthetic. Code: `tier2_confound_tests.py` (T5 section).

**Dimensionality scaling (Section 5.7):** 3D/5D/16D, 3 seeds, 30 epochs, 200 episodes synthetic. Code: `tier3_highdim.py`.

**Rico UI drift (Section 5.8):** Seed 42, 100 epochs, Rico dataset (500 samples). Code: `rico_drift_v2.ipynb`.

All analysis code, notebooks, data collection scripts, and results are available at https://github.com/revenue7-eng/prescribed-axes-drift.

---

## References

Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. CVPR 2023.

Balestriero, R., & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544.

Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR 2022.

Lazarev, A. (2026). The Space Matters More Than the Loss: JEPA Collapse as a Problem of Structure, Not Optimization.

LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. Technical Report, Meta AI.

Maes, L., Le Lidec, Q., Scieur, D., LeCun, Y., & Balestriero, R. (2026). LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. arXiv:2603.19312.

Ozkara, B., et al. (2025). PAN: A World Model for General, Interactable, and Long-Horizon World Simulation. arXiv:2511.09057.

Psenka, M., Rabbat, M., Krishnapriyan, A., LeCun, Y., & Bar, A. (2026). Parallel Stochastic Gradient-Based Planning for World Models. arXiv:2602.00475.
