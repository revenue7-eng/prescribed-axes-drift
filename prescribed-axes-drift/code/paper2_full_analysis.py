#!/usr/bin/env python3
"""
Paper 2 Full Analysis — v3 (robust saves)
==========================================
Saves to Google Drive after EVERY experiment block:
- after prescribed cov+drift
- after free cov+drift  
- after each freeze test
Maximum loss on disconnect: one freeze test (~2 min).

Usage:
  python paper2_v3.py --episodes 200 --epochs 30
  python paper2_v3.py --synthetic --episodes 200 --epochs 30
"""

import os, time, json, argparse, shutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# ================================================================
# Save helper — writes locally AND to Drive
# ================================================================
DRIVE_PATH = Path("/content/drive/MyDrive/paper2_results")
LOCAL_PATH = Path("paper2_results")

def save_results(data, filename="all_results.json"):
    LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    local_file = LOCAL_PATH / filename
    with open(local_file, "w") as f:
        json.dump(data, f, indent=2)
    try:
        DRIVE_PATH.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_file, DRIVE_PATH / filename)
        print(f"    [SAVED to Drive: {filename}]")
    except Exception as e:
        print(f"    [Drive save failed: {e}, local copy OK]")


# ================================================================
# Infrastructure
# ================================================================

class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=512):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots); dt = 3/(knots-1)
        w = torch.full((knots,), 2*dt); w[[0,-1]] = dt
        phi = torch.exp(-t.square()/2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", w*phi)
    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3)-self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights)*proj.size(-2)).mean()


def collect_gym_data(n_ep=200, max_steps=300, fs=5, seed=42):
    try:
        import gymnasium as gym
        import gym_pusht
    except ImportError:
        print("gym-pusht not found -> synthetic fallback")
        return collect_synthetic_data(n_ep, max_steps, fs, seed)
    rng = np.random.default_rng(seed)
    eps = []
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    for i in range(n_ep):
        obs, _ = env.reset(seed=int(rng.integers(0, 100000)))
        ss, aa = [obs.copy()], []
        for step in range(max_steps):
            if step == 0 or len(aa) == 0 or rng.random() < 0.3:
                a = env.action_space.sample()
            else:
                a = aa[-1] + rng.normal(0, 30, 2).astype(np.float32)
                a = np.clip(a, 0, 512)
            obs, _, d, tr, _ = env.step(a)
            if (step+1) % fs == 0:
                ss.append(obs.copy()); aa.append(a.copy())
            if d or tr: break
        if len(aa) >= 4:
            eps.append({"s": np.array(ss[:len(aa)+1]), "a": np.array(aa)})
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{n_ep}")
    env.close()
    print(f"Gym: {len(eps)} eps, avg {np.mean([len(e['a']) for e in eps]):.0f}")
    return eps

def collect_synthetic_data(n_ep=200, max_steps=300, fs=5, seed=42):
    rng = np.random.default_rng(seed); eps = []
    for _ in range(n_ep):
        ag = rng.uniform(50, 462, 2).astype(np.float32)
        bp = rng.uniform(100, 412, 2).astype(np.float32)
        ba = np.float32(rng.uniform(0, 2*np.pi))
        st = np.array([ag[0],ag[1],bp[0],bp[1],ba], dtype=np.float32)
        ss, aa = [st.copy()], []
        tgt = rng.uniform(50, 462, 2).astype(np.float32)
        for step in range(max_steps):
            if step % 20 == 0:
                tgt = rng.uniform(50, 462, 2).astype(np.float32)
            act = np.clip(tgt + rng.normal(0, 10, 2), 0, 512).astype(np.float32)
            d = act - ag; dn = np.linalg.norm(d)
            if dn > 0: ag += d * min(1.0, 20.0/dn)
            ag = np.clip(ag, 0, 512)
            tb = bp - ag; cd = np.linalg.norm(tb)
            if 0 < cd < 30:
                f = (30-cd)/30*5; bp += (tb/cd)*f
                ba = (ba + rng.normal(0, 0.05)*f) % (2*np.pi)
            bp = np.clip(bp, 0, 512)
            st = np.array([ag[0],ag[1],bp[0],bp[1],ba], dtype=np.float32)
            if (step+1) % fs == 0:
                ss.append(st.copy()); aa.append(act)
        if len(aa) >= 4:
            eps.append({"s": np.array(ss[:len(aa)+1]), "a": np.array(aa)})
    print(f"Synth: {len(eps)} eps, avg {np.mean([len(e['a']) for e in eps]):.0f}")
    return eps

class SeqDS(Dataset):
    def __init__(self, eps, H=3):
        self.w = []
        for e in eps:
            s, a = e["s"], e["a"]
            for t in range(len(a)-H):
                self.w.append((s[t:t+H+2].astype(np.float32),
                              a[t:t+H+1].astype(np.float32)))
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        s, a = self.w[i]
        return torch.from_numpy(s), torch.from_numpy(a)


class PrescEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("sc", torch.tensor([1/512, 1/512, 1/(2*np.pi)]))
    def forward(self, s): return s[..., 2:5] * self.sc

class FreeEnc(nn.Module):
    def __init__(self, di=5, do=3, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(di, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, do))
    def forward(self, s): return self.net(s)

class ActEnc(nn.Module):
    def __init__(self, di=2, do=3, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di, h), nn.GELU(), nn.Linear(h, do))
    def forward(self, a): return self.net(a)

class Pred(nn.Module):
    def __init__(self, d=3, H=3, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(H*2*d, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, d))
    def forward(self, e, ae):
        return self.net(torch.cat([e, ae], dim=-1).reshape(e.size(0), -1))

class Model(nn.Module):
    def __init__(self, enc, aenc, pred, sig, H=3):
        super().__init__()
        self.enc, self.aenc, self.pred, self.sig, self.H = enc, aenc, pred, sig, H
    def forward(self, s, a):
        H = self.H; emb = self.enc(s)
        ctx, tgt = emb[:, :H], emb[:, H]
        ae = self.aenc(a[:, :H]); p = self.pred(ctx, ae)
        return {"pl": F.mse_loss(p, tgt.detach()),
                "sl": self.sig(emb.transpose(0, 1)),
                "emb": emb.detach()}

def make_model(mode, seed):
    torch.manual_seed(seed)
    enc = PrescEnc() if mode == "prescribed" else FreeEnc(5, 3, 64)
    return Model(enc, ActEnc(2, 3, 32), Pred(3, 3, 128), SIGReg(17, 512), 3)


# ================================================================
# Analysis functions
# ================================================================

def train_epoch(mdl, tl, opt, lam, freeze_enc=False):
    mdl.train()
    if freeze_enc: mdl.enc.eval()
    tp, ts, n = 0, 0, 0
    for s, a in tl:
        o = mdl(s, a); l = o["pl"] + lam * o["sl"]
        opt.zero_grad(); l.backward()
        nn.utils.clip_grad_norm_(mdl.parameters(), 1.0); opt.step()
        b = s.size(0); tp += o["pl"].item()*b; ts += o["sl"].item()*b; n += b
    return tp/n, ts/n

@torch.no_grad()
def collect_embeddings(mdl, vl):
    mdl.eval(); all_emb = []; tp, n = 0, 0
    for s, a in vl:
        o = mdl(s, a); b = s.size(0)
        tp += o["pl"].item()*b; n += b
        all_emb.append(o["emb"][:, -1].cpu())
    return torch.cat(all_emb, 0).numpy(), tp/n

def cov_analysis(emb):
    cov = np.cov(emb - emb.mean(0), rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ep = ev[ev > 1e-12]; en = ep/ep.sum()
    er = np.exp(-np.sum(en * np.log(en + 1e-12)))
    cn = ep[0]/ep[-1] if ep[-1] > 1e-12 else float('inf')
    iso = ep[-1]/ep[0] if ep[0] > 1e-12 else 0
    return {"eig": ev.tolist(), "rank": float(er), "cond": float(cn),
            "iso": float(iso), "var": np.diag(cov).tolist()}

def procrustes(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    R = Vt.T @ U.T
    Y_aligned = Yc @ R + mx
    residual = np.sqrt(((X - Y_aligned)**2).sum(1).mean())
    return Y_aligned, residual

def gt_projection_r2(emb_t, emb_t1, gt):
    N = emb_t.shape[0]
    X_t = np.hstack([emb_t, np.ones((N, 1))])
    X_t1 = np.hstack([emb_t1, np.ones((emb_t1.shape[0], 1))])
    W, _, _, _ = np.linalg.lstsq(X_t, gt, rcond=None)
    def r2(pred, true):
        ss_res = ((true - pred)**2).sum()
        ss_tot = ((true - true.mean(0))**2).sum()
        return float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.0
    return r2(X_t @ W, gt), r2(X_t1 @ W, gt)


# ================================================================
# Experiment blocks
# ================================================================

def make_dataloaders(eps, seed, bs=128):
    ds = SeqDS(eps, 3)
    nt = int(len(ds)*0.9); nv = len(ds) - nt
    tr, va = random_split(ds, [nt, nv],
                          generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(tr, batch_size=bs, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=bs)
    # GT states
    gt = np.array([ds[va.indices[i]][0][-1, 2:5].numpy() for i in range(nv)])
    return tl, vl, gt


def run_cov_drift(mode, tl, vl, gt, seed, epochs, sample_epochs):
    """Covariance + Drift analysis for one mode."""
    print(f"\n  {mode.upper()} cov+drift...")
    mdl = make_model(mode, seed)
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                            lr=3e-4, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    prev_emb = None
    cov_snaps = []
    drift_snaps = []
    best, bep = float("inf"), 0
    
    for ep in range(epochs + 1):
        if ep > 0:
            train_epoch(mdl, tl, opt, 0.09)
            sch.step()
        
        emb, vp = collect_embeddings(mdl, vl)
        if ep > 0 and vp < best: best, bep = vp, ep
        
        if ep in sample_epochs:
            ca = cov_analysis(emb)
            cov_snaps.append({"epoch": ep, "val_loss": vp, **ca})
        
        if prev_emb is not None:
            raw = float(np.sqrt(((prev_emb - emb)**2).sum(1).mean()))
            _, aligned = procrustes(prev_emb, emb)
            r2_self, r2_xfer = gt_projection_r2(prev_emb, emb, gt)
            drift_snaps.append({
                "epoch_from": ep-1, "epoch_to": ep,
                "raw_drift": raw, "aligned_drift": float(aligned),
                "r2_self": r2_self, "r2_transfer": r2_xfer,
                "r2_drop": r2_self - r2_xfer, "val_loss": vp,
            })
        
        prev_emb = emb.copy()
        
        if ep % 10 == 0 or ep <= 3:
            print(f"    ep {ep:3d} val={vp:.6f}")
    
    print(f"    Best: {best:.6f} (ep {bep})")
    return {
        "mode": mode, "seed": seed, "best_vp": best, "best_ep": bep,
        "covariance": cov_snaps, "drift": drift_snaps,
    }


def run_freeze(mode, tl, vl, seed, epochs, freeze_at=None):
    """Single freeze test run."""
    label = mode if freeze_at is None else f"{mode}_freeze@{freeze_at}"
    
    mdl = make_model(mode, seed)
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                            lr=3e-4, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    encoder_frozen = False
    hist = []
    
    for ep in range(1, epochs + 1):
        if freeze_at is not None and ep == freeze_at + 1 and not encoder_frozen:
            for p in mdl.enc.parameters():
                p.requires_grad = False
            encoder_frozen = True
            opt = torch.optim.AdamW(
                [p for p in mdl.parameters() if p.requires_grad],
                lr=3e-4, weight_decay=1e-3)
        
        train_epoch(mdl, tl, opt, 0.09, freeze_enc=encoder_frozen)
        _, vp = collect_embeddings(mdl, vl)
        sch.step()
        hist.append({"ep": ep, "vp": vp})
    
    best = min(h["vp"] for h in hist)
    bep = min(hist, key=lambda h: h["vp"])["ep"]
    return {"label": label, "best_vp": best, "best_ep": bep, "hist": hist}


# ================================================================
# Main
# ================================================================

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--synthetic", action="store_true")
    pa.add_argument("--episodes", type=int, default=200)
    pa.add_argument("--epochs", type=int, default=30)
    pa.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    args = pa.parse_args()
    
    EPOCHS = args.epochs
    SEEDS = args.seeds
    sample_epochs = [e for e in [0,1,2,3,5,10,15,20,25,30,40,50] if e <= EPOCHS]
    freeze_points = [f for f in [1,2,3,5,7,10] if f < EPOCHS]
    
    collect = collect_synthetic_data if args.synthetic else collect_gym_data
    
    # Load existing results (resume support)
    results_file = LOCAL_PATH / "all_results.json"
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")
    else:
        all_results = {}
    
    all_results["config"] = {
        "episodes": args.episodes, "epochs": EPOCHS,
        "seeds": SEEDS, "synthetic": args.synthetic,
    }
    
    for seed in SEEDS:
        seed_key = f"seed_{seed}"
        
        # Skip if seed fully completed
        if seed_key in all_results:
            sr = all_results[seed_key]
            expected = 2 + 2 + len(freeze_points)  # 2 cov_drift + 2 freeze baselines + N freeze tests
            if len(sr) >= expected:
                print(f"\n  Seed {seed}: already complete, skipping")
                continue
        
        print(f"\n{'#'*60}\n  SEED {seed}\n{'#'*60}")
        
        torch.manual_seed(seed); np.random.seed(seed)
        eps = collect(args.episodes, 300, 5, seed)
        tl, vl, gt = make_dataloaders(eps, seed)
        
        if seed_key not in all_results:
            all_results[seed_key] = {}
        sr = all_results[seed_key]
        
        # --- Block 1: Prescribed cov+drift ---
        if "cov_drift_prescribed" not in sr:
            r = run_cov_drift("prescribed", tl, vl, gt, seed, EPOCHS, sample_epochs)
            sr["cov_drift_prescribed"] = r
            save_results(all_results)
        else:
            print(f"  prescribed cov+drift: cached")
        
        # --- Block 2: Free cov+drift ---
        if "cov_drift_free" not in sr:
            r = run_cov_drift("free", tl, vl, gt, seed, EPOCHS, sample_epochs)
            sr["cov_drift_free"] = r
            save_results(all_results)
        else:
            print(f"  free cov+drift: cached")
        
        # --- Block 3: Freeze prescribed ---
        if "freeze_prescribed" not in sr:
            print(f"\n  Freeze tests (seed={seed})")
            r = run_freeze("prescribed", tl, vl, seed, EPOCHS)
            sr["freeze_prescribed"] = r
            print(f"    prescribed: {r['best_vp']:.6f}")
            save_results(all_results)
        else:
            print(f"  freeze prescribed: cached ({sr['freeze_prescribed']['best_vp']:.6f})")
        
        # --- Block 4: Freeze free unfrozen ---
        if "freeze_free_unfrozen" not in sr:
            r = run_freeze("free", tl, vl, seed, EPOCHS)
            sr["freeze_free_unfrozen"] = r
            print(f"    free unfrozen: {r['best_vp']:.6f}")
            save_results(all_results)
        else:
            print(f"  freeze free unfrozen: cached ({sr['freeze_free_unfrozen']['best_vp']:.6f})")
        
        # --- Block 5+: Freeze at various points ---
        for T in freeze_points:
            key = f"freeze_free_at_{T}"
            if key not in sr:
                r = run_freeze("free", tl, vl, seed, EPOCHS, freeze_at=T)
                sr[key] = r
                print(f"    free freeze@{T}: {r['best_vp']:.6f}")
                save_results(all_results)
            else:
                print(f"  freeze@{T}: cached ({sr[key]['best_vp']:.6f})")
    
    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"{'SUMMARY':^70}")
    print(f"{'='*70}")
    
    for seed in SEEDS:
        seed_key = f"seed_{seed}"
        if seed_key not in all_results:
            continue
        sr = all_results[seed_key]
        
        p = sr.get("cov_drift_prescribed", {})
        f = sr.get("cov_drift_free", {})
        
        if p and f:
            pv = p["best_vp"]; fv = f["best_vp"]
            print(f"\n--- Seed {seed} ---")
            print(f"  Val loss: P={pv:.6f}  F={fv:.6f}  ratio={fv/pv:.1f}×")
        
        fp = sr.get("freeze_prescribed", {}).get("best_vp")
        fu = sr.get("freeze_free_unfrozen", {}).get("best_vp")
        if fp and fu:
            print(f"  Freeze: prescribed={fp:.6f}  unfrozen={fu:.6f}")
            for T in freeze_points:
                key = f"freeze_free_at_{T}"
                if key in sr:
                    fv = sr[key]["best_vp"]
                    imp = (fu - fv)/fu * 100
                    print(f"    freeze@{T}: {fv:.6f} ({imp:+.1f}% vs unfrozen)")
    
    print(f"\nDone. Results in {LOCAL_PATH}/all_results.json")


if __name__ == "__main__":
    main()
