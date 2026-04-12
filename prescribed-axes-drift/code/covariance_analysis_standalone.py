#!/usr/bin/env python3
"""
Covariance Analysis: Prescribed vs Free Latent Space
=====================================================
Based on lewm_pusht_experiment.py from prescribed-axes repo.

Trains both models, collects all validation embeddings,
computes covariance matrix, eigenspectrum, effective rank,
and per-axis variance over training epochs.

Question: Does prescribed axes preserve rank and spectral
structure where free space collapses?

Connection to GRASP (Theorem 1): if prescribed axes maintain
full rank, gradients remain well-conditioned — the adversarial
gradient problem doesn't arise because the space is anchored.

Connection to Mythos dual-role features: if collapse = loss of
connotation = loss of rank, then prescribed axes = axes with
fixed connotation = preserved rank.
"""

import os, time, json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


# --- SIGReg ---
class SIGReg(nn.Module):
    def __init__(self, knots=17, num_proj=512):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3/(knots-1)
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


# --- Data ---
def collect_synthetic_data(n_ep=200, max_steps=300, fs=5, seed=42):
    rng = np.random.default_rng(seed)
    eps = []
    for _ in range(n_ep):
        ag = rng.uniform(50,462,2).astype(np.float32)
        bp = rng.uniform(100,412,2).astype(np.float32)
        ba = np.float32(rng.uniform(0,2*np.pi))
        st = np.array([ag[0],ag[1],bp[0],bp[1],ba],dtype=np.float32)
        ss, aa = [st.copy()], []
        tgt = rng.uniform(50,462,2).astype(np.float32)
        for step in range(max_steps):
            if step%20==0: tgt = rng.uniform(50,462,2).astype(np.float32)
            act = np.clip(tgt+rng.normal(0,10,2),0,512).astype(np.float32)
            d = act-ag; dn = np.linalg.norm(d)
            if dn>0: ag += d*min(1.0,20.0/dn)
            ag = np.clip(ag,0,512)
            tb = bp-ag; cd = np.linalg.norm(tb)
            if 0<cd<30:
                f = (30-cd)/30*5; bp += (tb/cd)*f
                ba = (ba+rng.normal(0,0.05)*f)%(2*np.pi)
            bp = np.clip(bp,0,512)
            st = np.array([ag[0],ag[1],bp[0],bp[1],ba],dtype=np.float32)
            if (step+1)%fs==0: ss.append(st.copy()); aa.append(act)
        if len(aa)>=4:
            eps.append({"s":np.array(ss[:len(aa)+1]),"a":np.array(aa)})
    print(f"Synth: {len(eps)} eps, avg {np.mean([len(e['a']) for e in eps]):.0f}")
    return eps

class SeqDS(Dataset):
    def __init__(self, eps, H=3):
        self.w = []
        for e in eps:
            s,a = e["s"],e["a"]
            for t in range(len(a)-H):
                self.w.append((s[t:t+H+2].astype(np.float32), a[t:t+H+1].astype(np.float32)))
    def __len__(self): return len(self.w)
    def __getitem__(self, i):
        s,a = self.w[i]
        return torch.from_numpy(s), torch.from_numpy(a)


# --- Models ---
class PrescEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("sc", torch.tensor([1/512,1/512,1/(2*np.pi)]))
    def forward(self, s): return s[...,2:5]*self.sc

class FreeEnc(nn.Module):
    def __init__(self, di=5, do=3, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,do))
    def forward(self, s): return self.net(s)

class ActEnc(nn.Module):
    def __init__(self, di=2, do=3, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di,h),nn.GELU(),nn.Linear(h,do))
    def forward(self, a): return self.net(a)

class Pred(nn.Module):
    def __init__(self, d=3, H=3, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H*2*d,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,h),nn.LayerNorm(h),nn.GELU(),
                                 nn.Linear(h,d))
    def forward(self, e, ae):
        return self.net(torch.cat([e,ae],dim=-1).reshape(e.size(0),-1))

class Model(nn.Module):
    def __init__(self, enc, aenc, pred, sig, H=3):
        super().__init__()
        self.enc,self.aenc,self.pred,self.sig,self.H = enc,aenc,pred,sig,H
    def forward(self, s, a):
        H = self.H
        emb = self.enc(s)
        ctx, tgt = emb[:,:H], emb[:,H]
        ae = self.aenc(a[:,:H])
        p = self.pred(ctx, ae)
        return {"pl": F.mse_loss(p,tgt.detach()),
                "sl": self.sig(emb.transpose(0,1)),
                "p": p.detach(), "t": tgt.detach(),
                "emb": emb.detach()}


# --- Training ---
@dataclass
class Cfg:
    n_ep:int=200; max_steps:int=300; fs:int=5; seed:int=42
    dim:int=3; hid:int=128; H:int=3
    epochs:int=50; bs:int=128; lr:float=3e-4; wd:float=1e-3
    lam:float=0.09; split:float=0.9

def train_ep(m, dl, opt, lam, dev):
    m.train(); tp,ts,n = 0,0,0
    for s,a in dl:
        s,a = s.to(dev),a.to(dev)
        o = m(s,a); l = o["pl"]+lam*o["sl"]
        opt.zero_grad(); l.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        b=s.size(0); tp+=o["pl"].item()*b; ts+=o["sl"].item()*b; n+=b
    return tp/n,ts/n

@torch.no_grad()
def collect_embeddings(m, dl, dev):
    """Collect all embeddings from validation set."""
    m.eval()
    all_emb = []
    tp, ts, n = 0, 0, 0
    per_axis_mse = []
    for s, a in dl:
        s, a = s.to(dev), a.to(dev)
        o = m(s, a)
        b = s.size(0)
        tp += o["pl"].item()*b
        ts += o["sl"].item()*b
        n += b
        # Collect target embeddings (the state we're predicting)
        all_emb.append(o["emb"][:, -1].cpu())  # last timestep = target
        per_axis_mse.append((o["p"].cpu() - o["t"].cpu()).pow(2))
    
    emb = torch.cat(all_emb, dim=0)  # [N, D]
    per_axis = torch.cat(per_axis_mse, dim=0).mean(0).tolist()
    return emb.numpy(), tp/n, ts/n, per_axis


def covariance_analysis(emb, label=""):
    """Full covariance analysis of embedding matrix."""
    N, D = emb.shape
    
    # Center
    mean = emb.mean(axis=0)
    centered = emb - mean
    
    # Covariance matrix
    cov = np.cov(centered, rowvar=False)  # [D, D]
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]  # descending
    eigenvectors = eigenvectors[:, ::-1]
    
    # Effective rank (exponential of entropy of normalized eigenvalues)
    ev_pos = eigenvalues[eigenvalues > 1e-12]
    ev_norm = ev_pos / ev_pos.sum()
    entropy = -np.sum(ev_norm * np.log(ev_norm + 1e-12))
    effective_rank = np.exp(entropy)
    
    # Condition number
    if ev_pos[-1] > 1e-12:
        condition_number = ev_pos[0] / ev_pos[-1]
    else:
        condition_number = float('inf')
    
    # Variance per axis
    var_per_axis = np.diag(cov)
    
    # Isotropy score: ratio of min to max eigenvalue (1 = perfectly isotropic)
    isotropy = ev_pos[-1] / ev_pos[0] if ev_pos[0] > 1e-12 else 0.0
    
    # Off-diagonal magnitude (correlation between axes)
    off_diag = np.abs(cov - np.diag(np.diag(cov)))
    mean_off_diag = off_diag.sum() / (D*D - D) if D > 1 else 0.0
    
    result = {
        "N": N,
        "D": D,
        "eigenvalues": eigenvalues.tolist(),
        "effective_rank": float(effective_rank),
        "max_rank": D,
        "condition_number": float(condition_number),
        "isotropy": float(isotropy),
        "var_per_axis": var_per_axis.tolist(),
        "total_variance": float(np.trace(cov)),
        "mean_off_diagonal": float(mean_off_diag),
        "cov_matrix": cov.tolist(),
        "mean": mean.tolist(),
    }
    
    print(f"\n  [{label}] Covariance Analysis:")
    print(f"    Samples: {N}, Dims: {D}")
    print(f"    Eigenvalues: {[f'{v:.6f}' for v in eigenvalues]}")
    print(f"    Effective rank: {effective_rank:.3f} / {D}")
    print(f"    Condition number: {condition_number:.2f}")
    print(f"    Isotropy (min/max eig): {isotropy:.4f}")
    print(f"    Variance per axis: {[f'{v:.6f}' for v in var_per_axis]}")
    print(f"    Total variance: {np.trace(cov):.6f}")
    print(f"    Mean off-diagonal |cov|: {mean_off_diag:.6f}")
    
    return result


def run_with_covariance(mode, eps, cfg, dev, sample_epochs=None):
    """Train model and collect covariance analysis at specified epochs."""
    print(f"\n{'='*60}\n  {mode.upper()} — with covariance tracking\n{'='*60}")
    
    ds = SeqDS(eps, cfg.H)
    nt = int(len(ds)*cfg.split); nv = len(ds)-nt
    tr, va = random_split(ds, [nt,nv], generator=torch.Generator().manual_seed(cfg.seed))
    tl = DataLoader(tr, batch_size=cfg.bs, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=cfg.bs)
    print(f"  train={nt} val={nv}")
    
    enc = PrescEnc() if mode=="prescribed" else FreeEnc(5, cfg.dim, 64)
    mdl = Model(enc, ActEnc(2,cfg.dim,32), Pred(cfg.dim,cfg.H,cfg.hid),
                SIGReg(17,512), cfg.H).to(dev)
    np_ = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
    print(f"  params: {np_:,}")
    
    opt = torch.optim.AdamW([p for p in mdl.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    
    if sample_epochs is None:
        sample_epochs = [1, 5, 10, 25, 50]
    
    best, bep = float("inf"), 0
    hist = []
    cov_snapshots = []
    
    # Epoch 0 (before training)
    emb, vp, vs, axes = collect_embeddings(mdl, vl, dev)
    cov_result = covariance_analysis(emb, f"{mode} ep=0")
    cov_snapshots.append({"epoch": 0, "val_loss": vp, **cov_result})
    
    for ep in range(1, cfg.epochs+1):
        t0 = time.time()
        tp, ts = train_ep(mdl, tl, opt, cfg.lam, dev)
        emb, vp, vs, axes = collect_embeddings(mdl, vl, dev)
        sch.step()
        if vp < best: best, bep = vp, ep
        
        hist.append({"ep":ep, "tp":tp, "ts":ts, "vp":vp, "vs":vs, "ax":axes})
        
        if ep in sample_epochs or ep == cfg.epochs:
            cov_result = covariance_analysis(emb, f"{mode} ep={ep}")
            cov_snapshots.append({"epoch": ep, "val_loss": vp, **cov_result})
        
        if ep % 10 == 0 or ep == 1:
            ax = ", ".join(f"{v:.6f}" for v in axes)
            print(f"  ep {ep:3d} | tr {tp:.6f} | val {vp:.6f} | sig {vs:.4f} | [{ax}] | {time.time()-t0:.1f}s")
    
    print(f"  Best: {best:.6f} (ep {bep})")
    return {
        "mode": mode,
        "params": np_,
        "best": best,
        "bep": bep,
        "hist": hist,
        "covariance": cov_snapshots,
    }


def main():
    cfg = Cfg(n_ep=200, epochs=50, bs=128, seed=42, dim=3)
    dev = torch.device("cpu")
    print(f"Device: {dev}")
    print(f"Config: {json.dumps(asdict(cfg), indent=2)}")
    
    # Sample epochs for covariance snapshots
    sample_epochs = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]
    
    # Collect data
    print(f"\n{'='*60}\n  DATA (synthetic)\n{'='*60}")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    eps = collect_synthetic_data(cfg.n_ep, cfg.max_steps, cfg.fs, cfg.seed)
    
    # Run both modes with 3 seeds
    all_results = {}
    for seed in [42, 123, 777]:
        all_results[seed] = {}
        cfg_s = Cfg(n_ep=200, epochs=50, bs=128, seed=seed, dim=3)
        torch.manual_seed(seed); np.random.seed(seed)
        eps_s = collect_synthetic_data(cfg_s.n_ep, cfg_s.max_steps, cfg_s.fs, seed)
        
        for mode in ["prescribed", "free"]:
            torch.manual_seed(seed)
            all_results[seed][mode] = run_with_covariance(
                mode, eps_s, cfg_s, dev, sample_epochs)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"{'COVARIANCE ANALYSIS SUMMARY':^60}")
    print(f"{'='*60}")
    
    for seed in all_results:
        print(f"\n--- Seed {seed} ---")
        for mode in ["prescribed", "free"]:
            r = all_results[seed][mode]
            final_cov = r["covariance"][-1]
            print(f"  {mode:12s} | val_loss {r['best']:.6f} | "
                  f"eff_rank {final_cov['effective_rank']:.3f}/{final_cov['max_rank']} | "
                  f"cond {final_cov['condition_number']:.2f} | "
                  f"isotropy {final_cov['isotropy']:.4f}")
    
    # Rank evolution comparison
    print(f"\n{'='*60}")
    print(f"{'RANK EVOLUTION (seed=42)':^60}")
    print(f"{'='*60}")
    print(f"  {'Epoch':>5} | {'Prescribed':>12} {'rank':>6} | {'Free':>12} {'rank':>6}")
    print(f"  {'-'*5}-+-{'-'*12}-{'-'*6}-+-{'-'*12}-{'-'*6}")
    
    p_cov = all_results[42]["prescribed"]["covariance"]
    f_cov = all_results[42]["free"]["covariance"]
    
    p_epochs = {c["epoch"]: c for c in p_cov}
    f_epochs = {c["epoch"]: c for c in f_cov}
    
    for ep in sorted(set(list(p_epochs.keys()) + list(f_epochs.keys()))):
        p_er = p_epochs[ep]["effective_rank"] if ep in p_epochs else ""
        p_vl = p_epochs[ep]["val_loss"] if ep in p_epochs else ""
        f_er = f_epochs[ep]["effective_rank"] if ep in f_epochs else ""
        f_vl = f_epochs[ep]["val_loss"] if ep in f_epochs else ""
        
        p_str = f"{p_vl:>12.6f} {p_er:>6.3f}" if isinstance(p_er, float) else f"{'':>12} {'':>6}"
        f_str = f"{f_vl:>12.6f} {f_er:>6.3f}" if isinstance(f_er, float) else f"{'':>12} {'':>6}"
        print(f"  {ep:>5d} | {p_str} | {f_str}")
    
    # Save
    out = Path("covariance_results")
    out.mkdir(exist_ok=True)
    
    # Compact save (without full cov matrices for readability)
    compact = {}
    for seed in all_results:
        compact[seed] = {}
        for mode in ["prescribed", "free"]:
            r = all_results[seed][mode]
            compact[seed][mode] = {
                "best_val_loss": r["best"],
                "best_epoch": r["bep"],
                "params": r["params"],
                "covariance_evolution": [
                    {
                        "epoch": c["epoch"],
                        "val_loss": c["val_loss"],
                        "effective_rank": c["effective_rank"],
                        "condition_number": c["condition_number"],
                        "isotropy": c["isotropy"],
                        "eigenvalues": c["eigenvalues"],
                        "var_per_axis": c["var_per_axis"],
                        "total_variance": c["total_variance"],
                        "mean_off_diagonal": c["mean_off_diagonal"],
                    }
                    for c in r["covariance"]
                ],
            }
    
    with open(out / "covariance_analysis.json", "w") as fh:
        json.dump(compact, fh, indent=2)
    
    # Save full histories
    for seed in all_results:
        for mode in ["prescribed", "free"]:
            with open(out / f"history_{mode}_seed{seed}.json", "w") as fh:
                json.dump(all_results[seed][mode]["hist"], fh)
    
    print(f"\nResults saved to {out}/")
    print("Done.")


if __name__ == "__main__":
    main()
