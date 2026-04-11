#!/usr/bin/env python3
"""
Random Fixed Encoder Baseline — Critical Control Experiment
============================================================
Tests whether coordinate STABILITY alone (without semantic alignment)
improves learning compared to a free drifting encoder.

Four conditions:
1. prescribed    — GT coordinates (x,y,θ), stable + aligned
2. free          — learned MLP, drifting + misaligned  
3. random_fixed  — random linear projection 5→3, frozen from epoch 0
4. rotated_prescribed — random orthogonal rotation of (x,y,θ), stable + aligned (up to rotation)

Expected outcomes and what they prove:
- random_fixed >> free → stability alone helps (even without semantics)
- random_fixed << prescribed → alignment matters on top of stability
- random_fixed ≈ free → stability without semantics is worthless
- rotated_prescribed ≈ prescribed → alignment is rotation-invariant (it's about fixing, not interpreting)

This is THE experiment that separates the "privileged information" critique
from the "coordinate stability" hypothesis.

Usage:
    python random_fixed_encoder.py                    # runs all conditions, 3 seeds
    python random_fixed_encoder.py --colab            # saves to Google Drive
"""

import numpy as np, json, time, argparse
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# === Infrastructure (same as freeze_test_standalone.py) ===

class SIGReg(nn.Module):
    def __init__(s, k=17, np_=512):
        super().__init__(); s.np_ = np_
        t = torch.linspace(0, 3, k); dt = 3/(k-1)
        w = torch.full((k,), 2*dt); w[[0,-1]] = dt
        phi = torch.exp(-t.square()/2.0)
        s.register_buffer('t', t); s.register_buffer('phi', phi)
        s.register_buffer('weights', w*phi)
    def forward(s, proj):
        A = torch.randn(proj.size(-1), s.np_, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x = (proj @ A).unsqueeze(-1) * s.t
        err = (x.cos().mean(-3) - s.phi).square() + x.sin().mean(-3).square()
        return ((err @ s.weights) * proj.size(-2)).mean()


def synth(n, seed=42):
    """Generate n episodes of Push-T-like physics."""
    rng = np.random.default_rng(seed); eps = []
    for _ in range(n):
        ag = rng.uniform(50, 462, 2).astype(np.float32)
        bp = rng.uniform(100, 412, 2).astype(np.float32)
        ba = np.float32(rng.uniform(0, 2*np.pi))
        st = np.array([ag[0],ag[1],bp[0],bp[1],ba], dtype=np.float32)
        ss, aa = [st.copy()], []
        tgt = rng.uniform(50, 462, 2).astype(np.float32)
        for step in range(300):
            if step % 20 == 0:
                tgt = rng.uniform(50, 462, 2).astype(np.float32)
            act = np.clip(tgt + rng.normal(0, 10, 2), 0, 512).astype(np.float32)
            d = act - ag; dn = np.linalg.norm(d)
            if dn > 0: ag += d * min(1., 20./dn)
            ag = np.clip(ag, 0, 512)
            tb = bp - ag; cd = np.linalg.norm(tb)
            if 0 < cd < 30:
                f = (30-cd)/30*5; bp += (tb/cd)*f
                ba = (ba + rng.normal(0, .05)*f) % (2*np.pi)
            bp = np.clip(bp, 0, 512)
            st = np.array([ag[0],ag[1],bp[0],bp[1],ba], dtype=np.float32)
            if (step+1) % 5 == 0:
                ss.append(st.copy()); aa.append(act)
        if len(aa) >= 4:
            eps.append({'s': np.array(ss[:len(aa)+1]), 'a': np.array(aa)})
    return eps


class DS(Dataset):
    def __init__(s, eps, H=3):
        s.w = []
        for e in eps:
            st, a = e['s'], e['a']
            for t in range(len(a)-H):
                s.w.append((st[t:t+H+2].astype(np.float32),
                           a[t:t+H+1].astype(np.float32)))
    def __len__(s): return len(s.w)
    def __getitem__(s, i):
        st, a = s.w[i]
        return torch.from_numpy(st), torch.from_numpy(a)


# === Encoders ===

class PrescribedEncoder(nn.Module):
    """GT coordinates: extract (x_b, y_b, θ_b) and normalize."""
    def __init__(s):
        super().__init__()
        s.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(s, x): return x[..., 2:5] * s.sc


class FreeEncoder(nn.Module):
    """Learnable MLP 5→3."""
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(5, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 3))
    def forward(s, x): return s.net(x)


class RandomFixedEncoder(nn.Module):
    """Random linear projection 5→3, frozen from initialization.
    
    This is the critical control: stable coordinates with NO semantic alignment.
    If this beats free encoder → stability alone helps.
    If this loses to free encoder → stability without semantics is worthless.
    """
    def __init__(s, seed=0):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        W = torch.randn(5, 3, generator=rng)
        b = torch.randn(3, generator=rng) * 0.1
        # Register as buffers (not parameters) → never updated by optimizer
        s.register_buffer('W', W)
        s.register_buffer('b', b)
    
    def forward(s, x):
        return x @ s.W + s.b


class RotatedPrescribedEncoder(nn.Module):
    """GT coordinates rotated by a random orthogonal matrix.
    
    Stable + aligned (up to rotation). Tests whether the predictor
    can work with prescribed structure even when axes are not interpretable.
    If rotated ≈ prescribed → alignment is rotation-invariant.
    """
    def __init__(s, seed=0):
        super().__init__()
        s.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
        # Random orthogonal matrix via QR decomposition
        rng = torch.Generator().manual_seed(seed)
        A = torch.randn(3, 3, generator=rng)
        Q, _ = torch.linalg.qr(A)
        s.register_buffer('Q', Q)
    
    def forward(s, x):
        normed = x[..., 2:5] * s.sc
        return normed @ s.Q


# === Model components (same as original) ===

class ActionEncoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 3))
    def forward(s, a): return s.net(a)


class Predictor(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(18, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 3))
    def forward(s, e, ae):
        return s.net(torch.cat([e, ae], -1).reshape(e.size(0), -1))


class WorldModel(nn.Module):
    def __init__(s, enc, ae, pr, sig):
        super().__init__()
        s.enc, s.ae, s.pr, s.sig = enc, ae, pr, sig
    def forward(s, st, a):
        emb = s.enc(st)
        ctx, tgt = emb[:, :3], emb[:, 3]
        aem = s.ae(a[:, :3])
        p = s.pr(ctx, aem)
        return {
            'pl': F.mse_loss(p, tgt.detach()),
            'sl': s.sig(emb.transpose(0, 1)),
        }


# === Training ===

def train_one_epoch(mdl, tl, opt, lam):
    mdl.train()
    tp, ts, n = 0, 0, 0
    for s, a in tl:
        o = mdl(s, a)
        l = o['pl'] + lam * o['sl']
        opt.zero_grad()
        l.backward()
        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()
        b = s.size(0)
        tp += o['pl'].item() * b
        ts += o['sl'].item() * b
        n += b
    return tp/n, ts/n


@torch.no_grad()
def val_loss(mdl, vl):
    mdl.eval()
    tp, n = 0, 0
    for s, a in vl:
        o = mdl(s, a)
        tp += o['pl'].item() * s.size(0)
        n += s.size(0)
    return tp / n


def run_condition(encoder_type, eps, seed, epochs=30, lam=0.09):
    """Run one condition.
    
    encoder_type: 'prescribed', 'free', 'random_fixed', 'rotated_prescribed'
    """
    torch.manual_seed(seed); np.random.seed(seed)
    
    ds = DS(eps, 3)
    nt = int(len(ds) * 0.9); nv = len(ds) - nt
    tr, va = random_split(ds, [nt, nv],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=64)
    
    # Create encoder
    if encoder_type == 'prescribed':
        enc = PrescribedEncoder()
    elif encoder_type == 'free':
        enc = FreeEncoder()
    elif encoder_type == 'random_fixed':
        enc = RandomFixedEncoder(seed=seed + 1000)
    elif encoder_type == 'rotated_prescribed':
        enc = RotatedPrescribedEncoder(seed=seed + 2000)
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")
    
    mdl = WorldModel(enc, ActionEncoder(), Predictor(), SIGReg())
    
    # Only optimize parameters that require grad
    trainable = [p for p in mdl.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-3)
    
    hist = []
    for ep in range(1, epochs + 1):
        tp, ts = train_one_epoch(mdl, tl, opt, lam)
        vp = val_loss(mdl, vl)
        hist.append({'ep': ep, 'vp': vp, 'tp': tp})
        if ep in [1, 5, 10, 20, 30]:
            print(f"  [{encoder_type}] seed={seed} ep={ep}: val_loss={vp:.6f}")
    
    best_vp = min(h['vp'] for h in hist)
    return {
        'encoder': encoder_type,
        'seed': seed,
        'best_val_loss': best_vp,
        'final_val_loss': hist[-1]['vp'],
        'history': hist,
    }


# === Main ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--colab', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 777])
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    
    conditions = ['prescribed', 'free', 'random_fixed', 'rotated_prescribed']
    
    print("=" * 60)
    print("RANDOM FIXED ENCODER — CRITICAL CONTROL EXPERIMENT")
    print("=" * 60)
    print(f"Seeds: {args.seeds}")
    print(f"Episodes: {args.episodes}, Epochs: {args.epochs}")
    print(f"Conditions: {conditions}")
    print()
    
    results = {}
    
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        eps = synth(args.episodes, seed)
        print(f"Generated {len(eps)} episodes")
        
        for cond in conditions:
            t0 = time.time()
            res = run_condition(cond, eps, seed, args.epochs)
            dt = time.time() - t0
            
            key = f"{cond}_seed{seed}"
            results[key] = res
            print(f"  {cond}: best={res['best_val_loss']:.6f} "
                  f"final={res['final_val_loss']:.6f} ({dt:.1f}s)")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<25} {'Seed 42':>10} {'Seed 123':>10} {'Seed 777':>10} {'Mean':>10}")
    print("-" * 65)
    
    for cond in conditions:
        vals = []
        for seed in args.seeds:
            key = f"{cond}_seed{seed}"
            vals.append(results[key]['best_val_loss'])
        mean = np.mean(vals)
        row = f"{cond:<25}"
        for v in vals:
            row += f" {v:>10.6f}"
        row += f" {mean:>10.6f}"
        print(row)
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    means = {}
    for cond in conditions:
        vals = [results[f"{cond}_seed{s}"]['best_val_loss'] for s in args.seeds]
        means[cond] = np.mean(vals)
    
    print(f"\nprescribed:         {means['prescribed']:.6f}")
    print(f"rotated_prescribed: {means['rotated_prescribed']:.6f}")
    print(f"random_fixed:       {means['random_fixed']:.6f}")
    print(f"free:               {means['free']:.6f}")
    
    ratio_free_vs_prescribed = means['free'] / means['prescribed']
    ratio_random_vs_free = means['free'] / means['random_fixed'] if means['random_fixed'] > 0 else float('inf')
    ratio_random_vs_prescribed = means['random_fixed'] / means['prescribed'] if means['prescribed'] > 0 else float('inf')
    ratio_rotated_vs_prescribed = means['rotated_prescribed'] / means['prescribed'] if means['prescribed'] > 0 else float('inf')
    
    print(f"\nfree / prescribed:          {ratio_free_vs_prescribed:.1f}×")
    print(f"random_fixed / prescribed:  {ratio_random_vs_prescribed:.1f}×")
    print(f"rotated / prescribed:       {ratio_rotated_vs_prescribed:.1f}×")
    
    if means['random_fixed'] < means['free']:
        print("\n→ random_fixed < free: STABILITY ALONE HELPS (even without semantics)")
        stability_effect = (means['free'] - means['random_fixed']) / means['free'] * 100
        print(f"  Stability contribution: {stability_effect:.1f}% improvement")
    else:
        print("\n→ random_fixed ≥ free: stability without semantics does NOT help")
    
    if means['rotated_prescribed'] < means['random_fixed']:
        print("→ rotated_prescribed < random_fixed: ALIGNMENT MATTERS on top of stability")
    
    if ratio_rotated_vs_prescribed < 1.5:
        print("→ rotated ≈ prescribed: alignment is ROTATION-INVARIANT (fixing matters, not interpreting)")
    
    # Save results
    out_path = Path('random_fixed_results.json')
    if args.colab:
        out_path = Path('/content/drive/MyDrive/prescribed-axes/random_fixed_results.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump({
            'config': {'seeds': args.seeds, 'episodes': args.episodes, 'epochs': args.epochs},
            'results': {k: {kk: vv for kk, vv in v.items() if kk != 'history'} 
                       for k, v in results.items()},
            'means': means,
            'full_results': results,
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
