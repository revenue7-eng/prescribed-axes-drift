#!/usr/bin/env python3
"""
Freeze Test: Does stopping the drift improve prediction?
==========================================================
Three conditions:
1. prescribed — encoder fixed from start (baseline)
2. free_unfrozen — encoder trains throughout (standard)
3. free_frozen_at_T — encoder trains until epoch T, then frozen

If free_frozen < free_unfrozen → moving foundation is the cause
If free_frozen ≈ free_unfrozen → drift isn't the problem

We test multiple freeze points T to find the pattern.
"""

import numpy as np, json, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# === Infrastructure ===

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
        }


# === Training ===

def train_one_epoch(mdl, tl, opt, lam, freeze_encoder=False):
    mdl.train()
    if freeze_encoder:
        mdl.enc.eval()
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


def run_condition(mode, eps, seed, epochs, freeze_at=None):
    """
    mode: 'prescribed' or 'free'
    freeze_at: epoch at which to freeze encoder (None = never freeze)
    """
    label = mode if freeze_at is None else f'{mode}_freeze@{freeze_at}'
    torch.manual_seed(seed); np.random.seed(seed)

    ds = DS(eps, 3)
    nt = int(len(ds) * 0.9); nv = len(ds) - nt
    tr, va = random_split(ds, [nt, nv],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=64)

    enc = PE() if mode == 'prescribed' else FE()
    mdl = M(enc, AE(), PR(), SIGReg())

    # Track which params to optimize
    all_params = [p for p in mdl.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=1e-3)

    hist = []
    encoder_frozen = False

    for ep in range(1, epochs + 1):
        # Check if we should freeze encoder at this epoch
        if freeze_at is not None and ep == freeze_at + 1 and not encoder_frozen:
            # Freeze encoder parameters
            for p in mdl.enc.parameters():
                p.requires_grad = False
            encoder_frozen = True
            # Rebuild optimizer with only unfrozen params
            unfrozen = [p for p in mdl.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(unfrozen, lr=3e-4, weight_decay=1e-3)

        tp, ts = train_one_epoch(mdl, tl, opt, 0.09,
                                  freeze_encoder=encoder_frozen)
        vp = val_loss(mdl, vl)
        hist.append({'ep': ep, 'vp': vp, 'tp': tp})

    best_vp = min(h['vp'] for h in hist)
    best_ep = min((h for h in hist), key=lambda h: h['vp'])['ep']
    return {
        'label': label,
        'best_vp': best_vp,
        'best_ep': best_ep,
        'final_vp': hist[-1]['vp'],
        'hist': hist,
    }


# === Main ===

if __name__ == '__main__':
    seed = 42
    EPOCHS = 15
    torch.manual_seed(seed); np.random.seed(seed)
    eps = synth(50, seed)
    print(f'Data: {len(eps)} episodes')

    # Conditions
    conditions = []

    # 1. Prescribed (baseline)
    print('\n=== PRESCRIBED ===')
    r = run_condition('prescribed', eps, seed, EPOCHS)
    conditions.append(r)
    print(f'  best={r["best_vp"]:.6f} (ep {r["best_ep"]})')

    # 2. Free unfrozen (standard)
    print('\n=== FREE (unfrozen) ===')
    r = run_condition('free', eps, seed, EPOCHS)
    conditions.append(r)
    print(f'  best={r["best_vp"]:.6f} (ep {r["best_ep"]})')

    # 3. Free frozen at various epochs
    for T in [1, 2, 3, 5, 7, 10]:
        print(f'\n=== FREE (freeze at epoch {T}) ===')
        r = run_condition('free', eps, seed, EPOCHS, freeze_at=T)
        conditions.append(r)
        print(f'  best={r["best_vp"]:.6f} (ep {r["best_ep"]})')

    # === Summary ===
    print(f'\n{"="*60}')
    print(f'{"FREEZE TEST RESULTS":^60}')
    print(f'{"="*60}')
    print(f'  {"Condition":<25} {"Best val":>10} {"Best ep":>8} {"Final val":>10}')
    print(f'  {"-"*25} {"-"*10} {"-"*8} {"-"*10}')
    for c in conditions:
        print(f'  {c["label"]:<25} {c["best_vp"]:>10.6f} {c["best_ep"]:>8d} {c["final_vp"]:>10.6f}')

    # === Key comparison ===
    prescribed = conditions[0]['best_vp']
    unfrozen = conditions[1]['best_vp']
    frozen_results = {c['label']: c['best_vp'] for c in conditions[2:]}

    print(f'\n  Prescribed:    {prescribed:.6f}')
    print(f'  Free unfrozen: {unfrozen:.6f}')
    print(f'  Free frozen:')
    for label, vp in frozen_results.items():
        improvement = (unfrozen - vp) / unfrozen * 100
        vs_prescribed = vp / prescribed
        print(f'    {label:<25} {vp:.6f}  '
              f'({improvement:+.1f}% vs unfrozen, '
              f'{vs_prescribed:.1f}× vs prescribed)')

    # Save
    out = Path('freeze_results')
    out.mkdir(exist_ok=True)
    with open(out / 'freeze_test.json', 'w') as fh:
        json.dump({c['label']: {
            'best_vp': c['best_vp'],
            'best_ep': c['best_ep'],
            'final_vp': c['final_vp'],
            'hist': c['hist'],
        } for c in conditions}, fh, indent=2)
    print(f'\nSaved to {out}/freeze_test.json')
