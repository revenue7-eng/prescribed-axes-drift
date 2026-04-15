#!/usr/bin/env python3
"""
Tier 3: High-Dimensional Generalization Tests
==============================================
T9a: Push-T with 5D latent (prescribed = all 5 coords, no subspace selection)
T9b: Push-T with 16D latent (prescribed = engineered features, high-dim)

Tests whether the prescribed axes advantage generalizes beyond the 3D case
where prescribed = trivial identity on a subspace.

Run: python tier3_highdim.py
Results: tier3_results.json
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

SEEDS = [42, 123, 777]
EPISODES = 200
EPOCHS = 30
SAVE_FILE = Path("tier3_results.json")

# ================================================================
# Infrastructure
# ================================================================

class SIGReg(nn.Module):
    def __init__(self):
        super().__init__()
        t = torch.linspace(0, 3, 17); dt = 3 / 16
        w = torch.full((17,), 2 * dt); w[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', phi)
        self.register_buffer('weights', w * phi)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 512, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x = (proj @ A).unsqueeze(-1) * self.t
        err = (x.cos().mean(-3) - self.phi).square() + x.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


def synth(n, seed=42):
    rng = np.random.default_rng(seed); eps = []
    for _ in range(n):
        ag = rng.uniform(50, 462, 2).astype(np.float32)
        bp = rng.uniform(100, 412, 2).astype(np.float32)
        ba = np.float32(rng.uniform(0, 2 * np.pi))
        st = np.array([ag[0], ag[1], bp[0], bp[1], ba], dtype=np.float32)
        ss, aa = [st.copy()], []
        tgt = rng.uniform(50, 462, 2).astype(np.float32)
        for step in range(300):
            if step % 20 == 0:
                tgt = rng.uniform(50, 462, 2).astype(np.float32)
            act = np.clip(tgt + rng.normal(0, 10, 2), 0, 512).astype(np.float32)
            d = act - ag; dn = np.linalg.norm(d)
            if dn > 0: ag += d * min(1., 20. / dn)
            ag = np.clip(ag, 0, 512)
            tb = bp - ag; cd = np.linalg.norm(tb)
            if 0 < cd < 30:
                f = (30 - cd) / 30 * 5
                bp += (tb / cd) * f
                ba = (ba + rng.normal(0, .05) * f) % (2 * np.pi)
            bp = np.clip(bp, 0, 512)
            st = np.array([ag[0], ag[1], bp[0], bp[1], ba], dtype=np.float32)
            if (step + 1) % 5 == 0:
                ss.append(st.copy()); aa.append(act)
        if len(aa) >= 4:
            eps.append({'s': np.array(ss[:len(aa) + 1]), 'a': np.array(aa)})
    return eps


class SeqDS(Dataset):
    def __init__(self, eps, H=3):
        self.w = []
        for e in eps:
            st, a = e['s'], e['a']
            for t in range(len(a) - H):
                self.w.append((st[t:t + H + 2].astype(np.float32),
                               a[t:t + H + 1].astype(np.float32)))
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        st, a = self.w[i]
        return torch.from_numpy(st), torch.from_numpy(a)


# ================================================================
# Encoders — 5D experiments
# ================================================================

class Prescribed5D(nn.Module):
    """All 5 state coordinates, normalized. No subspace selection."""
    def __init__(self):
        super().__init__()
        self.register_buffer('sc', torch.tensor([
            1/512, 1/512, 1/512, 1/512, 1/(2*np.pi)
        ]))
    def forward(self, x): return x * self.sc


class Free5D(nn.Module):
    """MLP 5->64->64->5"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 5))
    def forward(self, x): return self.net(x)


class RandomFixed5Dto5D(nn.Module):
    """Random orthogonal 5->5, frozen. Stable, no alignment."""
    def __init__(self, seed=0):
        super().__init__()
        self.register_buffer('sc', torch.tensor([
            1/512, 1/512, 1/512, 1/512, 1/(2*np.pi)
        ]))
        rng = torch.Generator().manual_seed(seed)
        A = torch.randn(5, 5, generator=rng)
        Q, _ = torch.linalg.qr(A)
        self.register_buffer('Q', Q)
    def forward(self, x):
        return (x * self.sc) @ self.Q


# ================================================================
# Encoders — 16D experiments
# ================================================================

class Prescribed16D(nn.Module):
    """Engineered 16D features from 5D state.
    
    [x_a, y_a, x_b, y_b, sin(theta), cos(theta),
     x_a*x_b, y_a*y_b, dist_ab, dx, dy,
     x_a^2, y_a^2, x_b^2, y_b^2, theta_norm]
    
    All normalized to ~[0,1] range.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        xa = x[..., 0] / 512
        ya = x[..., 1] / 512
        xb = x[..., 2] / 512
        yb = x[..., 3] / 512
        theta = x[..., 4]
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        dx = xa - xb
        dy = ya - yb
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        
        features = torch.stack([
            xa, ya, xb, yb,                     # 0-3: positions
            sin_t, cos_t,                        # 4-5: angle components
            xa * xb, ya * yb,                    # 6-7: interactions
            dist,                                # 8: distance
            dx, dy,                              # 9-10: relative position
            xa**2, ya**2, xb**2, yb**2,          # 11-14: quadratics
            theta / (2 * np.pi),                 # 15: normalized angle
        ], dim=-1)
        return features


class Free16D(nn.Module):
    """MLP 5->128->128->16"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 16))
    def forward(self, x): return self.net(x)


class RandomFixed16D(nn.Module):
    """Random projection 5->16, frozen and normalized."""
    def __init__(self, seed=0):
        super().__init__()
        self.register_buffer('sc', torch.tensor([
            1/512, 1/512, 1/512, 1/512, 1/(2*np.pi)
        ]))
        rng = torch.Generator().manual_seed(seed)
        W = torch.randn(5, 16, generator=rng)
        # Normalize columns
        W = W / W.norm(dim=0, keepdim=True)
        self.register_buffer('W', W)

    def forward(self, x):
        return (x * self.sc) @ self.W


# ================================================================
# Model (parameterized by latent dim)
# ================================================================

class ActionEncoder(nn.Module):
    def __init__(self, d_latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.GELU(), nn.Linear(32, d_latent))
    def forward(self, a): return self.net(a)


class Predictor(nn.Module):
    def __init__(self, d_latent, H=3):
        super().__init__()
        d_in = H * 2 * d_latent
        h = max(128, d_latent * 8)
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, d_latent))
    def forward(self, e, ae):
        return self.net(torch.cat([e, ae], -1).reshape(e.size(0), -1))


class WorldModel(nn.Module):
    def __init__(self, enc, d_latent, H=3):
        super().__init__()
        self.enc = enc
        self.ae = ActionEncoder(d_latent)
        self.pr = Predictor(d_latent, H)
        self.sig = SIGReg()
        self.H = H

    def forward(self, st, a):
        emb = self.enc(st)
        ctx, tgt = emb[:, :self.H], emb[:, self.H]
        aem = self.ae(a[:, :self.H])
        p = self.pr(ctx, aem)
        return {
            'pl': F.mse_loss(p, tgt.detach()),
            'sl': self.sig(emb.transpose(0, 1)),
            'emb': emb.detach()
        }


def make_data(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    eps = synth(EPISODES, seed)
    ds = SeqDS(eps, 3)
    nt = int(len(ds) * 0.9); nv = len(ds) - nt
    tr, va = random_split(ds, [nt, nv],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=64)
    gt = np.array([ds[va.indices[i]][0][-1, 2:5].numpy() for i in range(nv)])
    return tl, vl, gt


@torch.no_grad()
def val_loss(mdl, vl):
    mdl.eval(); tp = 0; n = 0
    for s, a in vl:
        o = mdl(s, a); tp += o['pl'].item() * s.size(0); n += s.size(0)
    return tp / n


@torch.no_grad()
def get_emb(mdl, vl):
    mdl.eval(); embs = []
    for s, a in vl:
        o = mdl(s, a); embs.append(o['emb'][:, -1].cpu())
    return torch.cat(embs, 0).numpy()


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def train_epoch(mdl, tl, opt, lam=0.09):
    mdl.train()
    for s, a in tl:
        o = mdl(s, a)
        l = o['pl'] + lam * o['sl']
        opt.zero_grad(); l.backward()
        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()


def run_condition(enc, d_latent, tl, vl, gt, seed, label):
    """Train one condition, return metrics."""
    torch.manual_seed(seed)
    mdl = WorldModel(enc, d_latent)
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)

    best_vl = float('inf')
    prev_emb = None
    drift_01 = None
    r2_transfer_01 = None

    for ep in range(EPOCHS + 1):
        if ep > 0:
            train_epoch(mdl, tl, opt)

        vl_ = val_loss(mdl, vl) if ep > 0 else None
        if vl_ is not None and vl_ < best_vl:
            best_vl = vl_

        # Measure drift on epoch 0->1
        if ep <= 1:
            emb = get_emb(mdl, vl)
            if prev_emb is not None:
                drift_01 = float(np.sqrt(((prev_emb - emb)**2).sum(1).mean()))
                # R2 transfer (linear)
                N = prev_emb.shape[0]
                X0 = np.hstack([prev_emb, np.ones((N, 1))])
                X1 = np.hstack([emb, np.ones((emb.shape[0], 1))])
                W, _, _, _ = np.linalg.lstsq(X0, gt, rcond=None)
                r2_transfer_01 = r2(X1 @ W, gt)
            prev_emb = emb.copy()

    n_params = sum(p.numel() for p in mdl.enc.parameters() if p.requires_grad)
    print(f"    {label:<30} best={best_vl:.6f}  drift01={drift_01 if drift_01 else 0:.4f}  "
          f"R2xfer01={r2_transfer_01 if r2_transfer_01 else 1:.4f}  "
          f"enc_params={n_params}")

    return {
        'best_val_loss': best_vl,
        'drift_01': drift_01,
        'r2_transfer_01': r2_transfer_01,
        'enc_params': n_params,
    }


# ================================================================
# T9a: 5D latent space
# ================================================================

def run_t9a(seed):
    print(f"\n  T9a: 5D latent, seed={seed}")
    tl, vl, gt = make_data(seed)
    results = {}

    conditions = [
        ("prescribed_5d",    Prescribed5D(),                    5),
        ("free_5d",          Free5D(),                          5),
        ("random_fixed_5d",  RandomFixed5Dto5D(seed=seed+100),  5),
    ]

    for label, enc, d in conditions:
        res = run_condition(enc, d, tl, vl, gt, seed, label)
        results[label] = res

    return results


# ================================================================
# T9b: 16D latent space
# ================================================================

def run_t9b(seed):
    print(f"\n  T9b: 16D latent, seed={seed}")
    tl, vl, gt = make_data(seed)
    results = {}

    conditions = [
        ("prescribed_16d",    Prescribed16D(),                   16),
        ("free_16d",          Free16D(),                         16),
        ("random_fixed_16d",  RandomFixed16D(seed=seed+200),     16),
    ]

    for label, enc, d in conditions:
        res = run_condition(enc, d, tl, vl, gt, seed, label)
        results[label] = res

    return results


# ================================================================
# Also: 3D baseline for comparison on same data
# ================================================================

class Prescribed3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(self, x): return x[..., 2:5] * self.sc

class Free3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 3))
    def forward(self, x): return self.net(x)

def run_baseline_3d(seed):
    print(f"\n  Baseline 3D, seed={seed}")
    tl, vl, gt = make_data(seed)
    results = {}

    conditions = [
        ("prescribed_3d", Prescribed3D(), 3),
        ("free_3d",       Free3D(),       3),
    ]

    for label, enc, d in conditions:
        res = run_condition(enc, d, tl, vl, gt, seed, label)
        results[label] = res

    return results


# ================================================================
# Main
# ================================================================

def save(data):
    with open(SAVE_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  [Saved to {SAVE_FILE}]")


def main():
    print("=" * 65)
    print("TIER 3: HIGH-DIMENSIONAL GENERALIZATION")
    print(f"Seeds: {SEEDS}, Episodes: {EPISODES}, Epochs: {EPOCHS}")
    print("=" * 65)

    if SAVE_FILE.exists():
        with open(SAVE_FILE) as f:
            all_results = json.load(f)
        print(f"Resuming from {SAVE_FILE}")
    else:
        all_results = {
            'config': {'seeds': SEEDS, 'episodes': EPISODES, 'epochs': EPOCHS},
            'baseline_3d': {}, 'T9a_5d': {}, 'T9b_16d': {}
        }

    total_t0 = time.time()

    # --- Baseline 3D ---
    print("\n" + "#" * 65)
    print("# BASELINE: 3D (for comparison)")
    print("#" * 65)
    for seed in SEEDS:
        key = str(seed)
        if key in all_results.get('baseline_3d', {}):
            print(f"  3D seed={seed}: cached")
            continue
        res = run_baseline_3d(seed)
        all_results['baseline_3d'][key] = res
        save(all_results)

    # --- T9a: 5D ---
    print("\n" + "#" * 65)
    print("# T9a: 5D LATENT (prescribed = all 5 coords)")
    print("#" * 65)
    for seed in SEEDS:
        key = str(seed)
        if key in all_results.get('T9a_5d', {}):
            print(f"  5D seed={seed}: cached")
            continue
        res = run_t9a(seed)
        all_results['T9a_5d'][key] = res
        save(all_results)

    # --- T9b: 16D ---
    print("\n" + "#" * 65)
    print("# T9b: 16D LATENT (prescribed = engineered features)")
    print("#" * 65)
    for seed in SEEDS:
        key = str(seed)
        if key in all_results.get('T9b_16d', {}):
            print(f"  16D seed={seed}: cached")
            continue
        res = run_t9b(seed)
        all_results['T9b_16d'][key] = res
        save(all_results)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for block_name, block_key in [("3D Baseline", "baseline_3d"),
                                   ("5D (T9a)", "T9a_5d"),
                                   ("16D (T9b)", "T9b_16d")]:
        print(f"\n--- {block_name} ---")
        block = all_results.get(block_key, {})
        if not block:
            print("  (no data)")
            continue

        # Collect all conditions
        all_conds = set()
        for seed_data in block.values():
            all_conds.update(seed_data.keys())
        all_conds = sorted(all_conds)

        print(f"{'Condition':<30}", end="")
        for s in SEEDS:
            print(f"  seed_{s:>3}", end="")
        print("      mean    drift01  R2xfer01")
        print("-" * 100)

        for cond in all_conds:
            vals, drifts, r2s = [], [], []
            line = f"{cond:<30}"
            for seed in SEEDS:
                d = block.get(str(seed), {}).get(cond)
                if d:
                    v = d['best_val_loss']
                    vals.append(v)
                    line += f"  {v:>9.6f}"
                    if d.get('drift_01') is not None:
                        drifts.append(d['drift_01'])
                    if d.get('r2_transfer_01') is not None:
                        r2s.append(d['r2_transfer_01'])
                else:
                    line += "          -"
            if vals:
                line += f"  {np.mean(vals):>9.6f}"
                line += f"  {np.mean(drifts):>7.4f}" if drifts else "        -"
                line += f"  {np.mean(r2s):>8.4f}" if r2s else "         -"
            print(line)

        # Ratios
        presc_key = [c for c in all_conds if 'prescribed' in c]
        free_key = [c for c in all_conds if 'free' in c]
        if presc_key and free_key:
            p_vals = [block[str(s)][presc_key[0]]['best_val_loss'] for s in SEEDS if str(s) in block]
            f_vals = [block[str(s)][free_key[0]]['best_val_loss'] for s in SEEDS if str(s) in block]
            p_mean = np.mean(p_vals)
            f_mean = np.mean(f_vals)
            print(f"\n  prescribed/free ratio: {f_mean/p_mean:.1f}x")

    print(f"\nTotal time: {time.time()-total_t0:.0f}s")
    print(f"Results saved to {SAVE_FILE}")


if __name__ == '__main__':
    main()
