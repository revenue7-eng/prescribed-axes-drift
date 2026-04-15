#!/usr/bin/env python3
"""
Tier 2 Confound Tests for Paper 2: Semantic Drift
==================================================
T4: Aligned-drifting without SIGReg (does SIGReg destroy aligned init?)
T5: Optimizer state preservation in freeze test (is +20% from optimizer reset?)
T7: Random projection from 3D block coords (subspace knowledge vs alignment)

Run: python tier2_confound_tests.py
Results: tier2_results.json

200 episodes synthetic, 3 seeds, 30 epochs.
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
SAVE_FILE = Path("tier2_results.json")

# ================================================================
# Infrastructure (identical to tier1)
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


# === Encoders ===

class PrescribedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(self, x): return x[..., 2:5] * self.sc


class FreeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 3))
    def forward(self, x): return self.net(x)


class AlignedDriftingLinear(nn.Module):
    """Linear encoder initialized exactly as prescribed mapping, then trainable."""
    def __init__(self):
        super().__init__()
        # Initialize W to extract [x_b, y_b, theta_b] with normalization
        W = torch.zeros(5, 3)
        W[2, 0] = 1.0 / 512
        W[3, 1] = 1.0 / 512
        W[4, 2] = 1.0 / (2 * np.pi)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(torch.zeros(3))
    def forward(self, x): return x @ self.W + self.b


class RandomFixed3D(nn.Module):
    """Random orthogonal projection from 3 BLOCK coordinates (not all 5).
    
    This isolates: does knowing the right subspace matter,
    or is any stable 3D->3D mapping from block coords enough?
    """
    def __init__(self, seed=0):
        super().__init__()
        self.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
        rng = torch.Generator().manual_seed(seed)
        A = torch.randn(3, 3, generator=rng)
        Q, _ = torch.linalg.qr(A)
        self.register_buffer('Q', Q)
    def forward(self, x):
        normed = x[..., 2:5] * self.sc
        return normed @ self.Q


class RandomFixed5D(nn.Module):
    """Random projection from all 5 state dims (same as paper's random_fixed)."""
    def __init__(self, seed=0):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        W = torch.randn(5, 3, generator=rng)
        b = torch.randn(3, generator=rng) * 0.1
        self.register_buffer('W', W)
        self.register_buffer('b', b)
    def forward(self, x): return x @ self.W + self.b


class RotatedPrescribed(nn.Module):
    """GT coordinates rotated by random orthogonal matrix."""
    def __init__(self, seed=0):
        super().__init__()
        self.register_buffer('sc', torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
        rng = torch.Generator().manual_seed(seed)
        A = torch.randn(3, 3, generator=rng)
        Q, _ = torch.linalg.qr(A)
        self.register_buffer('Q', Q)
    def forward(self, x):
        return x[..., 2:5] * self.sc @ self.Q


# === Model ===

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
    def __init__(self, enc, use_sigreg=True):
        super().__init__()
        self.enc = enc
        self.ae = ActionEncoder()
        self.pr = Predictor()
        self.sig = SIGReg() if use_sigreg else None

    def forward(self, st, a):
        emb = self.enc(st)
        ctx, tgt = emb[:, :3], emb[:, 3]
        aem = self.ae(a[:, :3])
        p = self.pr(ctx, aem)
        result = {'pl': F.mse_loss(p, tgt.detach()), 'emb': emb.detach()}
        if self.sig is not None:
            result['sl'] = self.sig(emb.transpose(0, 1))
        else:
            result['sl'] = torch.tensor(0.0)
        return result


def make_data(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    eps = synth(EPISODES, seed)
    ds = SeqDS(eps, 3)
    nt = int(len(ds) * 0.9); nv = len(ds) - nt
    tr, va = random_split(ds, [nt, nv],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=64, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=64)
    return tl, vl


@torch.no_grad()
def val_loss(mdl, vl):
    mdl.eval(); tp = 0; n = 0
    for s, a in vl:
        o = mdl(s, a); tp += o['pl'].item() * s.size(0); n += s.size(0)
    return tp / n


def train_epoch(mdl, tl, opt, lam=0.09):
    mdl.train()
    for s, a in tl:
        o = mdl(s, a)
        l = o['pl'] + lam * o['sl']
        opt.zero_grad(); l.backward()
        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()


# ================================================================
# T4: Aligned-drifting without SIGReg
# ================================================================

def run_t4(seed):
    """Test whether SIGReg destroys aligned initialization."""
    print(f"\n  T4 seed={seed}")
    tl, vl = make_data(seed)
    results = {}

    conditions = [
        ("aligned_linear_with_sigreg", lambda: AlignedDriftingLinear(), True, 0.09),
        ("aligned_linear_no_sigreg",   lambda: AlignedDriftingLinear(), True, 0.0),
        ("free_with_sigreg",           lambda: FreeEncoder(),          True, 0.09),
        ("free_no_sigreg",             lambda: FreeEncoder(),          True, 0.0),
        ("prescribed",                 lambda: PrescribedEncoder(),    True, 0.09),
    ]

    for label, enc_fn, use_sig, lam in conditions:
        torch.manual_seed(seed)
        enc = enc_fn()
        mdl = WorldModel(enc, use_sigreg=(lam > 0))
        opt = torch.optim.AdamW(
            [p for p in mdl.parameters() if p.requires_grad],
            lr=3e-4, weight_decay=1e-3)

        best = float('inf')
        for ep in range(1, EPOCHS + 1):
            train_epoch(mdl, tl, opt, lam=lam)
            vl_ = val_loss(mdl, vl)
            if vl_ < best: best = vl_

        results[label] = best
        print(f"    {label:<35} {best:.6f}")

    return results


# ================================================================
# T5: Optimizer state preservation in freeze test
# ================================================================

def run_t5(seed):
    """Compare freeze with optimizer reset vs freeze preserving optimizer state."""
    print(f"\n  T5 seed={seed}")
    tl, vl = make_data(seed)
    results = {}

    # Prescribed baseline
    torch.manual_seed(seed)
    mdl = WorldModel(PrescribedEncoder())
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)
    best = float('inf')
    for ep in range(1, EPOCHS + 1):
        train_epoch(mdl, tl, opt)
        vl_ = val_loss(mdl, vl)
        if vl_ < best: best = vl_
    results['prescribed'] = best
    print(f"    prescribed: {best:.6f}")

    # Free unfrozen baseline
    torch.manual_seed(seed)
    mdl = WorldModel(FreeEncoder())
    opt = torch.optim.AdamW(
        [p for p in mdl.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=1e-3)
    best = float('inf')
    for ep in range(1, EPOCHS + 1):
        train_epoch(mdl, tl, opt)
        vl_ = val_loss(mdl, vl)
        if vl_ < best: best = vl_
    results['free_unfrozen'] = best
    print(f"    free unfrozen: {best:.6f}")

    # Freeze@1 with NEW optimizer (original method — potential confound)
    for freeze_at in [1, 3]:
        torch.manual_seed(seed)
        mdl = WorldModel(FreeEncoder())
        opt = torch.optim.AdamW(
            [p for p in mdl.parameters() if p.requires_grad],
            lr=3e-4, weight_decay=1e-3)

        best = float('inf')
        for ep in range(1, EPOCHS + 1):
            if ep == freeze_at + 1:
                # METHOD A: new optimizer (original paper method)
                for p in mdl.enc.parameters():
                    p.requires_grad = False
                opt = torch.optim.AdamW(
                    [p for p in mdl.parameters() if p.requires_grad],
                    lr=3e-4, weight_decay=1e-3)

            freeze_enc = (ep > freeze_at)
            if freeze_enc:
                mdl.enc.eval()
            train_epoch(mdl, tl, opt)
            vl_ = val_loss(mdl, vl)
            if vl_ < best: best = vl_

        results[f'freeze@{freeze_at}_new_opt'] = best
        print(f"    freeze@{freeze_at} NEW optimizer: {best:.6f}")

    # Freeze@1 PRESERVING optimizer state (remove encoder params from groups)
    for freeze_at in [1, 3]:
        torch.manual_seed(seed)
        mdl = WorldModel(FreeEncoder())

        # Track which params are encoder vs rest
        enc_param_ids = set(id(p) for p in mdl.enc.parameters())
        all_params = [p for p in mdl.parameters() if p.requires_grad]
        pred_params = [p for p in all_params if id(p) not in enc_param_ids]

        # Use single param group initially, but we'll need to manipulate later
        opt = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=1e-3)

        best = float('inf')
        frozen = False
        for ep in range(1, EPOCHS + 1):
            if ep == freeze_at + 1 and not frozen:
                # METHOD B: keep optimizer, just freeze encoder params
                for p in mdl.enc.parameters():
                    p.requires_grad = False
                frozen = True
                # Create new optimizer only for pred params, but seed with
                # matching LR/weight_decay to keep conditions fair.
                # The key difference: we DON'T reset momentum for pred params.
                # AdamW stores state per-param, so we rebuild optimizer but
                # copy the state for predictor params.
                old_state = opt.state
                opt_new = torch.optim.AdamW(pred_params, lr=3e-4, weight_decay=1e-3)
                # Copy optimizer state for predictor params
                for p in pred_params:
                    if p in old_state:
                        opt_new.state[p] = old_state[p]
                opt = opt_new

            if frozen:
                mdl.enc.eval()
            train_epoch(mdl, tl, opt)
            vl_ = val_loss(mdl, vl)
            if vl_ < best: best = vl_

        results[f'freeze@{freeze_at}_keep_state'] = best
        print(f"    freeze@{freeze_at} KEEP state:    {best:.6f}")

    return results


# ================================================================
# T7: Random projection from 3D block coords
# ================================================================

def run_t7(seed):
    """Compare random projections from 5D vs 3D subspace."""
    print(f"\n  T7 seed={seed}")
    tl, vl = make_data(seed)
    results = {}

    conditions = [
        ("prescribed",       lambda: PrescribedEncoder()),
        ("rotated_prescribed", lambda: RotatedPrescribed(seed=seed + 2000)),
        ("random_fixed_3d",  lambda: RandomFixed3D(seed=seed + 3000)),
        ("random_fixed_5d",  lambda: RandomFixed5D(seed=seed + 1000)),
        ("free",             lambda: FreeEncoder()),
    ]

    for label, enc_fn in conditions:
        torch.manual_seed(seed)
        enc = enc_fn()
        mdl = WorldModel(enc)
        opt = torch.optim.AdamW(
            [p for p in mdl.parameters() if p.requires_grad],
            lr=3e-4, weight_decay=1e-3)

        best = float('inf')
        for ep in range(1, EPOCHS + 1):
            train_epoch(mdl, tl, opt)
            vl_ = val_loss(mdl, vl)
            if vl_ < best: best = vl_

        results[label] = best
        print(f"    {label:<25} {best:.6f}")

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
    print("TIER 2 CONFOUND TESTS")
    print(f"Seeds: {SEEDS}, Episodes: {EPISODES}, Epochs: {EPOCHS}")
    print("=" * 65)

    if SAVE_FILE.exists():
        with open(SAVE_FILE) as f:
            all_results = json.load(f)
        print(f"Resuming from {SAVE_FILE}")
    else:
        all_results = {
            'config': {'seeds': SEEDS, 'episodes': EPISODES, 'epochs': EPOCHS},
            'T4': {}, 'T5': {}, 'T7': {}
        }

    total_t0 = time.time()

    # --- T4: Aligned-drifting without SIGReg ---
    print("\n" + "#" * 65)
    print("# T4: ALIGNED-DRIFTING WITHOUT SIGREG")
    print("# Does SIGReg destroy aligned initialization?")
    print("#" * 65)
    for seed in SEEDS:
        key = str(seed)
        if key in all_results.get('T4', {}):
            print(f"  T4 seed={seed}: cached")
            continue
        t0 = time.time()
        res = run_t4(seed)
        all_results['T4'][key] = res
        print(f"  T4 seed={seed}: done ({time.time()-t0:.0f}s)")
        save(all_results)

    # --- T5: Optimizer state preservation ---
    print("\n" + "#" * 65)
    print("# T5: OPTIMIZER STATE PRESERVATION IN FREEZE TEST")
    print("# Is the +20% from optimizer reset or from stability?")
    print("#" * 65)
    for seed in SEEDS:
        key = str(seed)
        if key in all_results.get('T5', {}):
            print(f"  T5 seed={seed}: cached")
            continue
        t0 = time.time()
        res = run_t5(seed)
        all_results['T5'][key] = res
        print(f"  T5 seed={seed}: done ({time.time()-t0:.0f}s)")
        save(all_results)

    # --- T7: Random projection from 3D block coords ---
    print("\n" + "#" * 65)
    print("# T7: RANDOM PROJECTION FROM 3D BLOCK COORDS")
    print("# Subspace knowledge vs alignment within subspace")
    print("#" * 65)
    for seed in SEEDS:
        key = str(seed)
        if key in all_results.get('T7', {}):
            print(f"  T7 seed={seed}: cached")
            continue
        t0 = time.time()
        res = run_t7(seed)
        all_results['T7'][key] = res
        print(f"  T7 seed={seed}: done ({time.time()-t0:.0f}s)")
        save(all_results)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    # T4
    print("\nT4: SIGReg effect on aligned initialization")
    conds4 = ['prescribed', 'aligned_linear_with_sigreg', 'aligned_linear_no_sigreg',
              'free_with_sigreg', 'free_no_sigreg']
    print(f"{'Condition':<38}", end="")
    for s in SEEDS: print(f"  seed_{s:>3}", end="")
    print("      mean")
    print("-" * 85)
    for cond in conds4:
        vals = []
        line = f"{cond:<38}"
        for seed in SEEDS:
            v = all_results['T4'].get(str(seed), {}).get(cond)
            if v is not None:
                vals.append(v)
                line += f"  {v:>9.6f}"
            else:
                line += "          -"
        if vals:
            line += f"  {np.mean(vals):>9.6f}"
        print(line)

    if all_results.get('T4'):
        s = str(SEEDS[0])
        d = all_results['T4'].get(s, {})
        if 'aligned_linear_with_sigreg' in d and 'aligned_linear_no_sigreg' in d:
            with_sig = np.mean([all_results['T4'][str(s)]['aligned_linear_with_sigreg'] for s in SEEDS])
            no_sig = np.mean([all_results['T4'][str(s)]['aligned_linear_no_sigreg'] for s in SEEDS])
            if with_sig > no_sig * 1.5:
                print(f"\n  → SIGReg makes aligned-drifting {with_sig/no_sig:.1f}x WORSE")
                print(f"    SIGReg actively destroys correct initialization")
            elif no_sig > with_sig * 1.5:
                print(f"\n  → SIGReg helps aligned-drifting by {no_sig/with_sig:.1f}x")
            else:
                print(f"\n  → SIGReg effect on aligned-drifting is small ({with_sig/no_sig:.2f}x)")

    # T5
    print("\nT5: Optimizer preservation in freeze test")
    conds5 = ['prescribed', 'free_unfrozen',
              'freeze@1_new_opt', 'freeze@1_keep_state',
              'freeze@3_new_opt', 'freeze@3_keep_state']
    print(f"{'Condition':<30}", end="")
    for s in SEEDS: print(f"  seed_{s:>3}", end="")
    print("      mean   vs_unfr")
    print("-" * 85)
    for cond in conds5:
        vals = []
        line = f"{cond:<30}"
        for seed in SEEDS:
            v = all_results['T5'].get(str(seed), {}).get(cond)
            if v is not None:
                vals.append(v)
                line += f"  {v:>9.6f}"
            else:
                line += "          -"
        if vals:
            m = np.mean(vals)
            unfr = np.mean([all_results['T5'].get(str(s), {}).get('free_unfrozen', m) for s in SEEDS])
            imp = (unfr - m) / unfr * 100 if unfr > 0 else 0
            line += f"  {m:>9.6f}  {imp:>+5.1f}%"
        print(line)

    # T7
    print("\nT7: Random projection subspace test")
    conds7 = ['prescribed', 'rotated_prescribed', 'random_fixed_3d',
              'random_fixed_5d', 'free']
    print(f"{'Condition':<25}", end="")
    for s in SEEDS: print(f"  seed_{s:>3}", end="")
    print("      mean   vs_free")
    print("-" * 85)
    means7 = {}
    for cond in conds7:
        vals = []
        line = f"{cond:<25}"
        for seed in SEEDS:
            v = all_results['T7'].get(str(seed), {}).get(cond)
            if v is not None:
                vals.append(v)
                line += f"  {v:>9.6f}"
            else:
                line += "          -"
        if vals:
            m = np.mean(vals)
            means7[cond] = m
            free_m = means7.get('free', m)
            ratio = free_m / m if m > 0 else 0
            line += f"  {m:>9.6f}  {ratio:>5.1f}x"
        print(line)

    if 'random_fixed_3d' in means7 and 'random_fixed_5d' in means7:
        r3 = means7['random_fixed_3d']
        r5 = means7['random_fixed_5d']
        print(f"\n  random_3d vs random_5d: {r3:.6f} vs {r5:.6f} ({r3/r5:.2f}x)")
        if r3 < r5 * 0.5:
            print("  → Knowing the right subspace provides additional advantage")
        elif r5 < r3 * 0.5:
            print("  → 5D projection better (more info from agent coords)")
        else:
            print("  → Subspace choice has limited effect vs stability")

    print(f"\nTotal time: {time.time()-total_t0:.0f}s")
    print(f"Results saved to {SAVE_FILE}")


if __name__ == '__main__':
    main()
