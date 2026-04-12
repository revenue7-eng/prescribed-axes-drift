#!/usr/bin/env python3
"""
Semantic Drift Analysis: Prescribed vs Free
=============================================
Three metrics:
1. Raw drift: mean ||f_t(x) - f_{t+1}(x)|| per epoch pair
2. Aligned drift: same after Procrustes alignment (removes rotation/scaling)
3. GT projection stability: linear probe h_t trained at epoch t,
   applied to f_{t+1} — does the mapping break?

If raw drift >> 0 but aligned drift ≈ 0 → just rotation, not semantic drift
If aligned drift >> 0 → structure changes, candidate for semantic drift
If GT projection degrades across epochs → semantics genuinely drifts
"""

import numpy as np, json, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# === Infrastructure (same as covariance_analysis) ===

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

class PE(nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(s, x): return x[..., 2:5] * s.sc

class FE(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(5, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 3))
    def forward(s, x): return s.net(x)

class AE(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 3))
    def forward(s, a): return s.net(a)

class PR(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(18, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 3))
    def forward(s, e, ae):
        return s.net(torch.cat([e, ae], -1).reshape(e.size(0), -1))

class M(nn.Module):
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
            'emb': emb.detach()
        }


# === Drift Metrics ===

def procrustes(X, Y):
    """Orthogonal Procrustes: find R minimizing ||X - Y@R||.
    Returns aligned Y and the residual."""
    # Center
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    # SVD of cross-covariance
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    R = (Vt.T @ U.T)
    Y_aligned = Yc @ R + mx
    residual = np.sqrt(((X - Y_aligned) ** 2).sum(1).mean())
    return Y_aligned, residual, R


def gt_projection_stability(emb_t, emb_t1, gt_t, gt_t1):
    """Train linear probe on (emb_t → gt_t), test on emb_t1 → gt_t1.
    Returns: R² on own epoch, R² transferred to next epoch."""
    # Fit linear regression: gt = emb @ W + b
    N, D = emb_t.shape
    # Add bias column
    X_t = np.hstack([emb_t, np.ones((N, 1))])
    X_t1 = np.hstack([emb_t1, np.ones((emb_t1.shape[0], 1))])

    # Least squares
    W, res, rank, sv = np.linalg.lstsq(X_t, gt_t, rcond=None)

    # Predict
    pred_self = X_t @ W
    pred_transfer = X_t1 @ W

    # R² 
    def r2(pred, true):
        ss_res = ((true - pred) ** 2).sum()
        ss_tot = ((true - true.mean(0)) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return r2(pred_self, gt_t), r2(pred_transfer, gt_t1)


# === Main ===

def collect_epoch_data(mdl, vl, val_states):
    """Collect embeddings and val loss for one epoch."""
    mdl.eval()
    all_emb = []; vl_sum = 0; n = 0
    with torch.no_grad():
        for s, a in vl:
            o = mdl(s, a)
            b = s.size(0)
            vl_sum += o['pl'].item() * b; n += b
            all_emb.append(o['emb'][:, -1])  # target embedding
    emb = torch.cat(all_emb).numpy()
    return emb, vl_sum / n


def run_drift_analysis(mode, eps, seed=42, epochs=15):
    print(f'\n{"="*60}\n  {mode.upper()} — Drift Analysis\n{"="*60}')
    torch.manual_seed(seed); np.random.seed(seed)

    ds = DS(eps, 3)
    nt = int(len(ds) * 0.9); nv = len(ds) - nt
    tr, va = random_split(ds, [nt, nv],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=64)

    # Collect ground truth states for validation set
    gt_states = []
    for idx in va.indices:
        s, a = ds[idx]
        # Target state = s[H+1] = s[4], extract (x, y, θ) = s[4, 2:5]
        gt = s[-1, 2:5].numpy()  # [block_x, block_y, block_angle]
        gt_states.append(gt)
    gt_states = np.array(gt_states)

    enc = PE() if mode == 'prescribed' else FE()
    mdl = M(enc, AE(), PR(), SIGReg())
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)

    # Collect embeddings at each epoch
    epoch_embeddings = []
    epoch_losses = []

    # Epoch 0 (before training)
    emb0, vl0 = collect_epoch_data(mdl, vl, gt_states)
    epoch_embeddings.append(emb0)
    epoch_losses.append(vl0)
    print(f'  ep  0: val={vl0:.6f}')

    for ep in range(1, epochs + 1):
        mdl.train()
        for s, a in tl:
            o = mdl(s, a)
            l = o['pl'] + 0.09 * o['sl']
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step()

        emb_ep, vl_ep = collect_epoch_data(mdl, vl, gt_states)
        epoch_embeddings.append(emb_ep)
        epoch_losses.append(vl_ep)
        if ep % 3 == 0 or ep == epochs:
            print(f'  ep {ep:2d}: val={vl_ep:.6f}')

    # === Compute metrics ===
    results = []
    for i in range(len(epoch_embeddings) - 1):
        E_t = epoch_embeddings[i]
        E_t1 = epoch_embeddings[i + 1]

        # 1. Raw drift
        raw = np.sqrt(((E_t - E_t1) ** 2).sum(1).mean())

        # 2. Procrustes-aligned drift
        _, aligned_residual, _ = procrustes(E_t, E_t1)

        # 3. GT projection stability
        r2_self, r2_transfer = gt_projection_stability(
            E_t, E_t1, gt_states, gt_states)

        results.append({
            'epoch_from': i,
            'epoch_to': i + 1,
            'val_loss_from': epoch_losses[i],
            'val_loss_to': epoch_losses[i + 1],
            'raw_drift': float(raw),
            'aligned_drift': float(aligned_residual),
            'r2_self': float(r2_self),
            'r2_transfer': float(r2_transfer),
            'r2_drop': float(r2_self - r2_transfer),
        })

    # Print summary
    print(f'\n  {"ep":>4} {"raw":>8} {"aligned":>8} {"R²self":>7} {"R²xfer":>7} {"R²drop":>7}')
    print(f'  {"-"*4} {"-"*8} {"-"*8} {"-"*7} {"-"*7} {"-"*7}')
    for r in results:
        print(f'  {r["epoch_from"]:>2}→{r["epoch_to"]:>1} '
              f'{r["raw_drift"]:>8.5f} {r["aligned_drift"]:>8.5f} '
              f'{r["r2_self"]:>7.4f} {r["r2_transfer"]:>7.4f} {r["r2_drop"]:>7.4f}')

    # Aggregates
    avg_raw = np.mean([r['raw_drift'] for r in results])
    avg_aligned = np.mean([r['aligned_drift'] for r in results])
    avg_r2_drop = np.mean([r['r2_drop'] for r in results])
    print(f'\n  Averages: raw={avg_raw:.5f}  aligned={avg_aligned:.5f}  R²drop={avg_r2_drop:.4f}')

    return {
        'mode': mode,
        'epochs': epochs,
        'per_epoch': results,
        'avg_raw_drift': float(avg_raw),
        'avg_aligned_drift': float(avg_aligned),
        'avg_r2_drop': float(avg_r2_drop),
        'final_val_loss': epoch_losses[-1],
    }


# === Entry point ===
if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed)
    eps = synth(50, seed)
    print(f'Data: {len(eps)} episodes')

    R = {}
    for mode in ['prescribed', 'free']:
        R[mode] = run_drift_analysis(mode, eps, seed=seed, epochs=15)

    # === Comparison ===
    print(f'\n{"="*60}')
    print(f'{"DRIFT COMPARISON":^60}')
    print(f'{"="*60}')
    print(f'  {"Metric":<25} {"Prescribed":>12} {"Free":>12} {"Ratio":>8}')
    print(f'  {"-"*25} {"-"*12} {"-"*12} {"-"*8}')

    p, f = R['prescribed'], R['free']
    for name, pk, fk in [
        ('Avg raw drift', 'avg_raw_drift', 'avg_raw_drift'),
        ('Avg aligned drift', 'avg_aligned_drift', 'avg_aligned_drift'),
        ('Avg R² drop', 'avg_r2_drop', 'avg_r2_drop'),
        ('Final val loss', 'final_val_loss', 'final_val_loss'),
    ]:
        pv, fv = p[pk], f[fk]
        ratio = fv / pv if abs(pv) > 1e-12 else float('inf')
        print(f'  {name:<25} {pv:>12.6f} {fv:>12.6f} {ratio:>8.1f}×')

    # Save
    out = Path('drift_results')
    out.mkdir(exist_ok=True)
    with open(out / 'drift_analysis.json', 'w') as fh:
        json.dump(R, fh, indent=2)
    print(f'\nSaved to {out}/drift_analysis.json')
