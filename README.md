# Semantic Drift, Not Rank Collapse

**Why Stable Coordinates Matter for Learning in Joint-Embedding Spaces**

Andrey Lazarev | Independent Researcher | April 2026

→ [**Paper draft (v8)**](reports/paper_draft_v8.md)

---

## Key Result

Representational collapse in JEPA is not a rank problem — it is a coordinate stability problem.

A free encoder with SIGReg maintains full rank (2.99/3) and high isotropy (0.86) — yet loses by **222×** to a prescribed encoder with lower rank and isotropy.

The drift has **two phases:**
- **Phase 1 (epochs 0–2):** information genuinely destroyed — neither linear nor MLP decoders can recover it
- **Phase 2 (epochs 3+):** information preserved but in nonlinearly drifting coordinates

Standard remedies fail:

| Intervention | Gap vs prescribed |
|---|---|
| Free encoder (baseline) | 222× |
| Differential LR (encoder 100× slower) | 62× |
| Extra predictor steps (K=5) | 171× |
| EMA target encoder | 6.1× |
| PCA canonicalization | worsens |
| **Prescribed axes** | **1.0×** |

### Subspace, Not Alignment

Any frozen, normalized basis in the task-relevant subspace matches prescribed performance:

| Condition | Val loss | vs free |
|---|---|---|
| Prescribed (x, y, θ) | 0.000037 | 222× |
| Rotated prescribed (random rotation) | 0.000039 | 212× |
| **Random fixed 3D (random orth. in block subspace)** | **0.000036** | **230×** |
| Free (drifting) | 0.008282 | baseline |

The predictor does not need axes aligned with physical coordinates. It needs a coordinate system that is (a) fixed, (b) normalized, and (c) spans the task-relevant subspace.

### Scaling with Dimensionality

| Dim | Gap prescribed/free | Drift (ep 0→1) | R² transfer |
|-----|--------------------|----|-------------|
| 3D | 169× | 1.53 | −70 |
| 5D | 66× | 1.91 | −65 |
| 16D | 50× | 3.58 | −596 |

Gap persists at all dimensions. Drift amplifies with dimensionality.

### The 2×2 Factorial Design

|  | **Stable** | **Unstable** |
|---|---|---|
| **Aligned** | 0.000039 (prescribed) | 0.012849 (aligned-drifting) |
| **Not aligned** | 0.000473 (random fixed 5D) | 0.008282 (free) |

Stability is a prerequisite: an encoder initialized on *perfect* coordinates but allowed to drift performs **no better than free**.

[![Main Figure](figures/figure1_main.png)](figures/figure1_main.png)

---

## Relation to Paper 1

This work extends [The Space Matters More Than the Loss](https://github.com/revenue7-eng/prescribed-axes) (Lazarev, 2026), which shows prescribed axes outperform free encoders across 4 modalities (5–38×). This paper identifies *why*: the free encoder's coordinate system drifts so rapidly that downstream modules cannot learn in it.

---

## Reproducing

### New experiments (v8)

**Two-phase drift + optimization lag + PCA:**
```bash
python code/tier1_all_tests.py    # T1: MLP decoder transfer, T2: update ratio, T3: PCA
```

**Confound tests (optimizer, SIGReg, subspace):**
```bash
python code/tier2_confound_tests.py   # T4: ±SIGReg, T5: optimizer state, T7: random 3D vs 5D
```

**Dimensionality scaling (3D / 5D / 16D):**
```bash
python code/tier3_highdim.py
```

### Original experiments

**Drift + Freeze + Covariance:**
1. Upload `code/paper2_colab.ipynb` to Colab
2. Run all cells (~20 min, CPU)

**Random Fixed Encoder control:**
1. Upload `code/paper2_random_fixed_v2_colab.ipynb` to Colab
2. Run all cells (~15 min, CPU)

**Aligned-but-Drifting (factorial cell):**
1. Upload `code/paper2_aligned_drifting_colab.ipynb` to Colab
2. Run all cells (~20 min, CPU)

**LR Sweep + EMA:**
1. Upload `code/lr_sweep_ema_baseline.ipynb` to Colab (T4 GPU)

**Rico UI Cross-Modal:**
1. Upload `code/rico_drift_v2.ipynb` to Colab (T4 GPU)

### Local

```bash
pip install torch numpy scipy matplotlib
python code/paper2_full_analysis.py      # drift + freeze + covariance
python code/random_fixed_encoder.py      # random fixed control
```

### Configuration

| Parameter | Value |
|---|---|
| Seeds | 42, 123, 777 |
| Episodes | 200 per seed (synthetic Push-T physics) |
| Epochs | 30 |
| SIGReg λ | 0.09 |
| Free encoder | MLP 5→64→64→3 |
| Prescribed | normalize(x_b, y_b, θ_b) |

---

## Repository Structure

```
README.md                                  ← This file
paper.pdf                                 ← Paper (v7 PDF, v8 in reports/)

code/
├── tier1_all_tests.py                    ← NEW: MLP decoder, update ratio, PCA
├── tier2_confound_tests.py               ← NEW: ±SIGReg, optimizer, random 3D
├── tier3_highdim.py                      ← NEW: 3D / 5D / 16D scaling
├── paper2_colab.ipynb                    ← Colab: drift + freeze + covariance
├── paper2_random_fixed_v2_colab.ipynb    ← Colab: random fixed encoder
├── paper2_aligned_drifting_colab.ipynb   ← Colab: aligned-but-drifting
├── lr_sweep_ema_baseline.ipynb           ← Colab: LR sweep + EMA
├── rico_drift_v2.ipynb                   ← Colab: Rico UI cross-modal drift
├── paper2_full_analysis.py               ← Complete analysis pipeline
├── drift_analysis_standalone.py          ← Drift: raw, Procrustes, R² transfer
├── covariance_analysis_standalone.py     ← Eigenspectrum, rank, isotropy
├── freeze_test_standalone.py             ← Freeze test
├── random_fixed_encoder.py              ← Random fixed encoder
└── generate_figures.py                  ← Reproduce all figures

results/
├── tier1_results.json                    ← NEW: MLP decoder, update ratio, PCA (3 seeds)
├── tier2_results.json                    ← NEW: confound tests (3 seeds)
├── tier3_results.json                    ← NEW: 3D/5D/16D scaling (3 seeds)
├── p2_dim_sweep_results.json             ← NEW: dim sweep 1–11 (3 seeds)
├── all_results.json                      ← Drift, covariance, freeze (3 seeds × 30 epochs)
├── random_fixed_v2_results.json          ← Random fixed control (4 conditions × 3 seeds)
├── aligned_drifting_results.json         ← Aligned-but-drifting (5 conditions × 3 seeds)
├── lr_sweep_results.json                 ← LR sweep + EMA (seed 42, 50 epochs)
└── rico_drift_v2_results.json            ← Rico UI drift (seed 42, 100 epochs)

reports/
├── paper_draft_v8.md                     ← Current draft (v8, with Tier 1-3 results)
├── paper_draft_v7.md                     ← Previous draft (v7)
├── paper_draft_v6.md                     ← Earlier draft (v6)
├── paper_draft_v5.md                     ← Earlier draft (v5)
├── paper_draft_v5_ru.md                  ← Russian translation (v5)
├── paper_v6.docx                         ← Word document (v6)
└── paper_v5.docx                         ← Word document (v5)

figures/
├── figure1_main.png                      ← Main 2×2 figure (300 dpi)
├── figure1_main.svg                      ← Main figure (vector)
└── ...                                   ← Individual figures
```

---

## Summary of Evidence (v8)

| Claim | Evidence | Section |
|---|---|---|
| Rank ≠ performance | rank 2.99, isotropy 0.86 → 222× worse | 3 |
| Coordinates drift | R² transfer < −62 after 1 epoch | 4.2 |
| Phase 1: info destroyed | MLP decoder R² = −283 on epoch 0→1 | 4.3 |
| Phase 2: nonlinear preservation | MLP R² = 0.81, linear R² = 0.69 (late epochs) | 4.3 |
| PCA cannot fix drift | PCA worsens R² transfer | 4.4 |
| Drift harms learning | freeze@1 → +20% (no optimizer confound, <3%) | 5.1–5.2 |
| Subspace, not alignment | random_3d ≈ prescribed (0.97×) | 5.4 |
| Alignment without stability = worthless | aligned-drifting ≈ free | 5.5 |
| Not optimization lag | diffLR 100× → 62× gap remains | 5.6 |
| Scales with dimension | 3D: 169×, 5D: 66×, 16D: 50× | 5.7 |
| Cross-modal | Rico UI vision: same drift pattern | 5.8 |

### What is established
- Full rank is not sufficient for good predictions
- Drift has two phases: early catastrophic, late nonlinear
- Drift is nonlinear (PCA fails) and structural (diffLR 100× leaves 62× gap)
- The advantage is subspace fixation, not axis alignment (random_3d ≈ prescribed)
- Stability is a prerequisite; alignment without it is worthless
- Effect persists from 3D to 16D

### What remains open
- Generalization beyond Push-T (double pendulum confirmed with normalization; vision pilot only)
- Behavior at very high data volume (500 episodes: free surpasses prescribed — see Paper 1)
- Soft freeze schedules vs hard freeze

---

## Citation

```bibtex
@article{lazarev2026drift,
  title={Semantic Drift, Not Rank Collapse: Why Stable Coordinates Matter
         for Learning in Joint-Embedding Spaces},
  author={Lazarev, Andrey},
  year={2026},
  url={https://github.com/revenue7-eng/prescribed-axes-drift}
}
```
