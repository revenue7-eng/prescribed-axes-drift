#!/usr/bin/env python3
"""
Tier 1 Critical Tests for Paper 2: Semantic Drift
==================================================
T1: MLP decoder transfer (does drift destroy information or just linear readability?)
T2: Update ratio / differential LR (is this optimization lag or structural?)
T3: PCA canonicalization (is drift rotation/scaling or nonlinear deformation?)

Run: python tier1_all_tests.py
Results: tier1_results.json

200 episodes synthetic, 3 seeds, 30 epochs.
Estimated time: ~40-60 min on CPU (Windows).
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

SEEDS = [42, 123, 777]
EPISODES = 200
EPOCHS = 30
SAVE_FILE = Path("tier1_results.json")

# ================================================================
# Infrastructure (identical to paper2 repo)
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


class FreeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 3))
    def forward(self, x): return self.net(x)


class PrescribedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(self, x): return x[..., 2:5] * self.sc


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, 3))
    def forward(self, a): return self.net(a)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 3))
    def forward(self, e, ae):
        return self.net(torch.cat([e, ae], -1).reshape(e.size(0), -1))


class WorldModel(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.ae = ActionEncoder()
        self.pr = Predictor()
        self.sig = SIGReg()

    def forward(self, st, a):
        emb = self.enc(st)
        ctx, tgt = emb[:, :3], emb[:, 3]
        aem = self.ae(a[:, :3])
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
def get_emb(mdl, vl):
    mdl.eval(); embs = []
    for s, a in vl:
        o = mdl(s, a); embs.append(o['emb'][:, -1].cpu())
    return torch.cat(embs, 0).numpy()


@torch.no_grad()
def val_loss(mdl, vl):
    mdl.eval(); tp = 0; n = 0
    for s, a in vl:
        o = mdl(s, a); tp += o['pl'].item() * s.size(0); n += s.size(0)
    return tp / n


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def train_epoch(mdl, tl, opt):
    mdl.train()
    for s, a in tl:
        o = mdl(s, a)
        l = o['pl'] + 0.09 * o['sl']
        opt.zero_grad(); l.backward()
        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()


# ================================================================
# T1: MLP Decoder Transfer
# ================================================================

def linear_transfer(emb_t, emb_t1, gt):
    N = emb_t.shape[0]
    X0 = np.hstack([emb_t, np.ones((N, 1))])
    X1 = np.hstack([emb_t1, np.ones((emb_t1.shape[0], 1))])
    W, _, _, _ = np.linalg.lstsq(X0, gt, rcond=None)
    return r2(X0 @ W, gt), r2(X1 @ W, gt)


def mlp_transfer(emb_t, emb_t1, gt, hidden=128, steps=1500, lr=1e-3):
    """Train MLP decoder on (emb_t -> gt), evaluate on emb_t1."""
    mu, sd = gt.mean(0), gt.std(0) + 1e-8
    gn = (gt - mu) / sd  # normalize targets

    dec = nn.Sequential(
        nn.Linear(3, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        nn.Linear(hidden, 3))
    opt = torch.optim.Adam(dec.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    X0 = torch.tensor(emb_t, dtype=torch.float32)
    Y = torch.tensor(gn, dtype=torch.float32)
    X1 = torch.tensor(emb_t1, dtype=torch.float32)

    dec.train()
    for _ in range(steps):
        p = dec(X0)
        loss = F.mse_loss(p, Y)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()

    dec.eval()
    with torch.no_grad():
        p0 = dec(X0).numpy() * sd + mu
        p1 = dec(X1).numpy() * sd + mu
    return r2(p0, gt), r2(p1, gt)


def run_t1(seed):
    print(f"\n  T1 seed={seed}")
    tl, vl, gt = make_data(seed)

    torch.manual_seed(seed)
    mdl = WorldModel(FreeEncoder())
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)

    results = []
    prev_emb = None

    for ep in range(EPOCHS + 1):
        if ep > 0:
            train_epoch(mdl, tl, opt)
        emb = get_emb(mdl, vl)

        if prev_emb is not None:
            ls, lx = linear_transfer(prev_emb, emb, gt)
            ms, mx = mlp_transfer(prev_emb, emb, gt)
            row = {
                'epoch_from': ep - 1, 'epoch_to': ep,
                'lin_self': round(ls, 5), 'lin_xfer': round(lx, 5),
                'mlp_self': round(ms, 5), 'mlp_xfer': round(mx, 5),
            }
            results.append(row)
            print(f"    {ep-1}->{ep:>2}  lin={lx:>8.4f}  mlp={mx:>8.4f}  "
                  f"{'MLP WINS' if mx > lx + 0.05 else 'both' if mx < lx - 0.05 else 'similar'}")

        prev_emb = emb.copy()

    return results


# ================================================================
# T2: Update Ratio / Differential LR
# ================================================================

def run_t2_condition(tl, vl, seed, label, enc_lr=3e-4, extra_pred_steps=0):
    """Train one condition and return best val loss."""
    torch.manual_seed(seed)
    mdl = WorldModel(FreeEncoder())

    if extra_pred_steps == 0 and enc_lr == 3e-4:
        # Standard training
        opt = torch.optim.AdamW(
            [p for p in mdl.parameters() if p.requires_grad],
            lr=3e-4, weight_decay=1e-3)
        best = float('inf')
        for ep in range(1, EPOCHS + 1):
            train_epoch(mdl, tl, opt)
            vl_ = val_loss(mdl, vl)
            if vl_ < best: best = vl_
        return best

    # Differential LR or extra predictor steps
    enc_params = list(mdl.enc.parameters())
    enc_ids = set(id(p) for p in enc_params)
    pred_params = [p for p in mdl.parameters()
                   if p.requires_grad and id(p) not in enc_ids]

    if enc_lr != 3e-4:
        opt_all = torch.optim.AdamW([
            {'params': enc_params, 'lr': enc_lr},
            {'params': pred_params, 'lr': 3e-4}
        ], weight_decay=1e-3)
    else:
        opt_all = torch.optim.AdamW(
            [p for p in mdl.parameters() if p.requires_grad],
            lr=3e-4, weight_decay=1e-3)

    opt_pred = torch.optim.AdamW(pred_params, lr=3e-4, weight_decay=1e-3)

    best = float('inf')
    for ep in range(1, EPOCHS + 1):
        mdl.train()
        for s, a in tl:
            # Joint step
            o = mdl(s, a); l = o['pl'] + 0.09 * o['sl']
            opt_all.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt_all.step()

            # Extra predictor-only steps
            if extra_pred_steps > 0:
                for p in enc_params: p.requires_grad_(False)
                for _ in range(extra_pred_steps):
                    o2 = mdl(s, a); l2 = o2['pl'] + 0.09 * o2['sl']
                    opt_pred.zero_grad(); l2.backward()
                    nn.utils.clip_grad_norm_(pred_params, 1.0)
                    opt_pred.step()
                for p in enc_params: p.requires_grad_(True)

        vl_ = val_loss(mdl, vl)
        if vl_ < best: best = vl_
    return best


def run_t2(seed):
    print(f"\n  T2 seed={seed}")
    tl, vl, gt = make_data(seed)

    # Prescribed baseline
    torch.manual_seed(seed)
    mdl_p = WorldModel(PrescribedEncoder())
    opt_p = torch.optim.AdamW(
        [p for p in mdl_p.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)
    best_p = float('inf')
    for ep in range(1, EPOCHS + 1):
        train_epoch(mdl_p, tl, opt_p)
        vl_ = val_loss(mdl_p, vl)
        if vl_ < best_p: best_p = vl_
    print(f"    prescribed: {best_p:.6f}")

    results = {'prescribed': best_p}

    # Free K=1 (baseline)
    f1 = run_t2_condition(tl, vl, seed, 'free_K1')
    results['free_K1'] = f1
    print(f"    free K=1:   {f1:.6f}  ({f1/best_p:.1f}x)")

    # Free K=3
    f3 = run_t2_condition(tl, vl, seed, 'free_K3', extra_pred_steps=2)
    results['free_K3'] = f3
    print(f"    free K=3:   {f3:.6f}  ({f3/best_p:.1f}x)")

    # Free K=5
    f5 = run_t2_condition(tl, vl, seed, 'free_K5', extra_pred_steps=4)
    results['free_K5'] = f5
    print(f"    free K=5:   {f5:.6f}  ({f5/best_p:.1f}x)")

    # Differential LR: encoder 10x slower
    fd = run_t2_condition(tl, vl, seed, 'free_diffLR_10x', enc_lr=3e-5)
    results['free_diffLR_10x'] = fd
    print(f"    diffLR 10x: {fd:.6f}  ({fd/best_p:.1f}x)")

    # Differential LR: encoder 100x slower
    fd100 = run_t2_condition(tl, vl, seed, 'free_diffLR_100x', enc_lr=3e-6)
    results['free_diffLR_100x'] = fd100
    print(f"    diffLR 100x:{fd100:.6f}  ({fd100/best_p:.1f}x)")

    return results


# ================================================================
# T3: PCA Canonicalization
# ================================================================

def pca_canonical(emb):
    mu = emb.mean(0)
    centered = emb - mu
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    # Sign convention: largest abs element positive
    for i in range(eigvecs.shape[1]):
        if eigvecs[np.argmax(np.abs(eigvecs[:, i])), i] < 0:
            eigvecs[:, i] *= -1
    return centered @ eigvecs


def run_t3(seed):
    print(f"\n  T3 seed={seed}")
    tl, vl, gt = make_data(seed)

    torch.manual_seed(seed)
    mdl = WorldModel(FreeEncoder())
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)

    results = []
    prev_emb = None
    prev_pca = None

    for ep in range(EPOCHS + 1):
        if ep > 0:
            train_epoch(mdl, tl, opt)
        emb = get_emb(mdl, vl)
        pca_emb = pca_canonical(emb)

        if prev_emb is not None:
            # Raw transfer
            N = prev_emb.shape[0]
            X0 = np.hstack([prev_emb, np.ones((N, 1))])
            X1 = np.hstack([emb, np.ones((emb.shape[0], 1))])
            W, _, _, _ = np.linalg.lstsq(X0, gt, rcond=None)
            raw_xfer = r2(X1 @ W, gt)

            # PCA canonical transfer
            P0 = np.hstack([prev_pca, np.ones((N, 1))])
            P1 = np.hstack([pca_emb, np.ones((pca_emb.shape[0], 1))])
            Wp, _, _, _ = np.linalg.lstsq(P0, gt, rcond=None)
            pca_xfer = r2(P1 @ Wp, gt)

            row = {
                'epoch_from': ep - 1, 'epoch_to': ep,
                'raw_xfer': round(raw_xfer, 5),
                'pca_xfer': round(pca_xfer, 5),
            }
            results.append(row)

            if ep <= 5 or ep % 5 == 0:
                print(f"    {ep-1}->{ep:>2}  raw={raw_xfer:>8.4f}  pca={pca_xfer:>8.4f}  "
                      f"{'PCA helps' if pca_xfer > raw_xfer + 0.05 else 'PCA neutral/worse'}")

        prev_emb = emb.copy()
        prev_pca = pca_emb.copy()

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
    print("TIER 1 CRITICAL TESTS")
    print(f"Seeds: {SEEDS}, Episodes: {EPISODES}, Epochs: {EPOCHS}")
    print("=" * 65)

    # Load existing results for resume
    if SAVE_FILE.exists():
        with open(SAVE_FILE) as f:
            all_results = json.load(f)
        print(f"Resuming from {SAVE_FILE}")
    else:
        all_results = {
            'config': {'seeds': SEEDS, 'episodes': EPISODES, 'epochs': EPOCHS},
            'T1': {}, 'T2': {}, 'T3': {}
        }

    total_t0 = time.time()

    # --- T1: MLP Decoder Transfer ---
    print("\n" + "#" * 65)
    print("# T1: MLP DECODER TRANSFER")
    print("#" * 65)
    for seed in SEEDS:
        key = str(seed)
        if key in all_results.get('T1', {}):
            print(f"  T1 seed={seed}: cached")
            continue
        t0 = time.time()
        res = run_t1(seed)
        all_results['T1'][key] = res
        print(f"  T1 seed={seed}: done ({time.time()-t0:.0f}s)")
        save(all_results)

    # --- T2: Update Ratio ---
    print("\n" + "#" * 65)
    print("# T2: UPDATE RATIO / DIFFERENTIAL LR")
    print("#" * 65)
    for seed in SEEDS:
        key = str(seed)
        if key in all_results.get('T2', {}):
            print(f"  T2 seed={seed}: cached")
            continue
        t0 = time.time()
        res = run_t2(seed)
        all_results['T2'][key] = res
        print(f"  T2 seed={seed}: done ({time.time()-t0:.0f}s)")
        save(all_results)

    # --- T3: PCA Canonicalization ---
    print("\n" + "#" * 65)
    print("# T3: PCA CANONICALIZATION")
    print("#" * 65)
    for seed in SEEDS:
        key = str(seed)
        if key in all_results.get('T3', {}):
            print(f"  T3 seed={seed}: cached")
            continue
        t0 = time.time()
        res = run_t3(seed)
        all_results['T3'][key] = res
        print(f"  T3 seed={seed}: done ({time.time()-t0:.0f}s)")
        save(all_results)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    # T1 summary
    print("\nT1: MLP Decoder Transfer")
    print(f"{'Trans':>6} ", end="")
    for seed in SEEDS:
        print(f"  lin_{seed}  mlp_{seed}", end="")
    print()
    # Show epochs 0->1, 1->2, 2->3, 5->6, 9->10
    for show_ep in [0, 1, 2, 5, 9, 14, 29]:
        line = f"{show_ep}->{show_ep+1:>2} "
        for seed in SEEDS:
            key = str(seed)
            data = all_results['T1'].get(key, [])
            found = False
            for row in data:
                if row['epoch_from'] == show_ep:
                    line += f"  {row['lin_xfer']:>7.3f}  {row['mlp_xfer']:>7.3f}"
                    found = True
                    break
            if not found:
                line += "       -        -"
        print(line)

    # T2 summary
    print("\nT2: Update Ratio / Differential LR")
    print(f"{'Condition':<20}", end="")
    for seed in SEEDS:
        print(f"  seed {seed:>3}", end="")
    print("     mean    ratio")

    for cond in ['prescribed', 'free_K1', 'free_K3', 'free_K5',
                 'free_diffLR_10x', 'free_diffLR_100x']:
        vals = []
        line = f"{cond:<20}"
        for seed in SEEDS:
            key = str(seed)
            v = all_results['T2'].get(key, {}).get(cond)
            if v is not None:
                vals.append(v)
                line += f"  {v:>8.6f}"
            else:
                line += "         -"
        if vals:
            mean = np.mean(vals)
            p_vals = [all_results['T2'].get(str(s), {}).get('prescribed', 1)
                      for s in SEEDS]
            p_mean = np.mean([v for v in p_vals if v is not None])
            ratio = mean / p_mean if p_mean > 0 else 0
            line += f"  {mean:>8.6f}  {ratio:>5.1f}x"
        print(line)

    # T3 summary
    print("\nT3: PCA Canonicalization")
    print(f"{'Trans':>6} ", end="")
    for seed in SEEDS:
        print(f"  raw_{seed}  pca_{seed}", end="")
    print()
    for show_ep in [0, 1, 2, 5, 9, 14, 29]:
        line = f"{show_ep}->{show_ep+1:>2} "
        for seed in SEEDS:
            key = str(seed)
            data = all_results['T3'].get(key, [])
            found = False
            for row in data:
                if row['epoch_from'] == show_ep:
                    line += f"  {row['raw_xfer']:>7.3f}  {row['pca_xfer']:>7.3f}"
                    found = True
                    break
            if not found:
                line += "       -        -"
        print(line)

    print(f"\nTotal time: {time.time()-total_t0:.0f}s")
    print(f"Results saved to {SAVE_FILE}")


if __name__ == '__main__':
    main()
