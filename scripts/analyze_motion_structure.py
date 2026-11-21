"""Analyse the motion structure of a trained Shape of Motion checkpoint.

Reports (doc "Relational Motion Priors", sections 5, 6, 9):
  * motion-weight statistics (entropy, max weight, dominant bases)
  * kNN TV of the weights
  * neighbour distance variation (rigidity) over the whole video
  * bones (connected components of near-equal weights), joints between
    adjacent bones (class, residual, pivot, hinge axis), static contacts
  * merge scores for adjacent bone pairs
and writes a bone-coloured point cloud (.ply) plus a JSON summary.

Usage (run on the idle GPU so it never competes with training):
  CUDA_VISIBLE_DEVICES=1 python scripts/analyze_motion_structure.py \
      --ckpt outputs/paper-windmill/checkpoints/last.ckpt --out outputs/paper-windmill/analysis
"""

import json
import os
import os.path as osp
import sys
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import tyro

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from flow3d.configs import PriorConfig  # noqa: E402
from flow3d.relational_prior import (  # noqa: E402
    JOINT_NAMES,
    STATIC,
    RelationalPrior,
    bone_mean_weights,
    charbonnier_tv,
    merge_score,
    robust_extent,
)
from flow3d.scene_model import SceneModel  # noqa: E402


@dataclass
class Args:
    ckpt: str
    out: str
    prior: PriorConfig = field(default_factory=PriorConfig)
    opaque_thresh: float = 0.5
    num_sample: int = 3000
    knn_rigidity: int = 8
    max_merge_points: int = 2000
    seed: int = 0


@torch.no_grad()
def fg_trajectories(model: SceneModel, chunk: int = 32) -> torch.Tensor:
    """(N, T, 3) world positions of all fg Gaussians."""
    T = model.num_frames
    outs = []
    for s in range(0, T, chunk):
        ts = torch.arange(s, min(s + chunk, T), device=model.fg.params["means"].device)
        m, _ = model.compute_poses_fg(ts)
        outs.append(m)
    return torch.cat(outs, dim=1)


@torch.no_grad()
def distance_variation(traj, idx_pool, num_sample, k, frame, gen):
    """Doc section 9: k nearest (within idx_pool) of sampled Gaussians at
    `frame`; std/mean of each pair's distance over all frames."""
    pool = traj[idx_pool]  # (P, T, 3)
    sel = torch.randperm(pool.shape[0], generator=gen)[:num_sample].to(pool.device)
    d = torch.cdist(pool[sel, frame], pool[:, frame])
    nn = d.topk(k + 1, largest=False).indices[:, 1:]  # (S, k)
    a = pool[sel][:, None]  # (S, 1, T, 3)
    b = pool[nn]  # (S, k, T, 3)
    dist = (a - b).norm(dim=-1)  # (S, k, T)
    var = dist.std(-1) / dist.mean(-1).clamp_min(1e-9)
    var = var.reshape(-1)
    return {
        "frame": int(frame),
        "median_std_over_mean": float(var.median()),
        "frac_gt_20pct": float((var > 0.2).float().mean()),
        "frac_gt_5pct": float((var > 0.05).float().mean()),
    }


def write_ply(path, xyz, rgb):
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")


@torch.no_grad()
def main(args: Args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = SceneModel.init_from_state_dict(ck["model"]).to(device)
    model.training = False
    T = model.num_frames
    gen = torch.Generator().manual_seed(args.seed)
    report: dict = {"ckpt": args.ckpt, "epoch": ck.get("epoch"), "num_frames": T}

    # ---- weights
    w = model.fg.get_coefs()
    op = model.fg.get_opacities()
    opaque = op > args.opaque_thresh
    mx, am = w.max(-1)
    ent = -(w * w.clamp_min(1e-9).log()).sum(-1)
    B = w.shape[1]
    hist = torch.bincount(am, minlength=B)
    report["weights"] = {
        "num_fg": int(w.shape[0]),
        "num_opaque": int(opaque.sum()),
        "num_bases": B,
        "mean_max_weight_all": float(mx.mean()),
        "mean_max_weight_opaque": float(mx[opaque].mean()) if opaque.any() else None,
        "frac_max_gt_0.9": float((mx > 0.9).float().mean()),
        "mean_entropy": float(ent.mean()),
        "uniform_entropy": float(np.log(B)),
        "argmax_hist": hist.tolist(),
        "bases_dominant_gt_1000": int((hist > 1000).sum()),
    }

    # ---- kNN graph + TV (same graph the prior uses)
    prior = RelationalPrior(args.prior, T)
    prior._maybe_rebuild_knn(model, epoch=0)
    e = prior.edges
    l1 = (w[e[:, 0]] - w[e[:, 1]]).abs().sum(-1)
    report["knn"] = {
        "k": args.prior.tv_k,
        "num_edges": int(e.shape[0]),
        "median_knn_dist": prior.knn_med,
        "fg_scale": prior.fg_scale,
        "tv_charbonnier": float(charbonnier_tv(w, e, args.prior.tv_eps)),
        "median_l1_weight_diff": float(l1.median()),
        "frac_edges_cut_at_tau": float((l1 > args.prior.bone_tau).float().mean()),
    }

    # ---- rigidity
    traj = fg_trajectories(model)  # (N, T, 3)
    pool = torch.nonzero(opaque).squeeze(-1)
    if pool.numel() < args.knn_rigidity + 1:
        pool = torch.arange(w.shape[0], device=device)
    report["rigidity"] = [
        distance_variation(traj, pool, args.num_sample, args.knn_rigidity, f, gen)
        for f in (0, T // 2)
    ]

    # ---- bones + joints
    bones = prior.rebuild_bones(model)
    prior.bones = bones
    sizes = bones.sizes.tolist()
    report["bones"] = {
        "num_bones": bones.num_bones,
        "num_valid": int(bones.valid.sum()),
        "largest_sizes": sorted(sizes, reverse=True)[:20],
        "frac_boundary": float(bones.boundary.float().mean()),
        "tau": args.prior.bone_tau,
        "n_min": args.prior.bone_n_min,
    }
    joints = []
    if bones.pairs.shape[0] > 0:
        sol, res_n, cls = prior.joint_terms(model)
        wbar = bone_mean_weights(w, bones.labels, bones.num_bones)
        tf = model.motion_bases.compute_transforms(torch.arange(T, device=device), wbar)
        means = model.fg.params["means"]
        for k, (a, b) in enumerate(bones.pairs.tolist()):
            row = {
                "a": a,
                "b": "static" if b == STATIC else b,
                "size_a": sizes[a],
                "size_b": None if b == STATIC else sizes[b],
                "contact_count": int(bones.pair_counts[k]),
                "class": JOINT_NAMES[int(cls[k])],
                "res_norm": float(res_n[k]),
                "eig_H0_over_T": [float(x) for x in sol.evals[k]],
                "pivot": [float(x) for x in sol.p[k]],
                "hinge_axis": [float(x) for x in sol.n1[k]],
            }
            if b != STATIC:
                ib = torch.nonzero(bones.labels == b).squeeze(-1)
                if ib.numel() > args.max_merge_points:
                    ib = ib[torch.randperm(ib.numel(), generator=gen)[: args.max_merge_points].to(device)]
                diam = 2 * robust_extent(means[bones.labels == b])
                row["merge_score"] = merge_score(
                    tf[a, :, :, :3], tf[a, :, :, 3], traj[ib], diam
                )
            joints.append(row)
    report["joints"] = joints
    report["joint_class_counts"] = {
        n: sum(1 for j in joints if j["class"] == n) for n in JOINT_NAMES
    }

    # ---- outputs
    rng = np.random.default_rng(0)
    palette = rng.integers(40, 255, size=(max(bones.num_bones, 1), 3))
    lab = bones.labels.cpu().numpy()
    rgb = palette[lab]
    rgb[~bones.valid.cpu().numpy()[lab]] = 80  # grey = too small for joints
    keep = (op > 0.1).cpu().numpy()
    write_ply(osp.join(args.out, "bones_canonical.ply"),
              model.fg.params["means"].cpu().numpy()[keep], rgb[keep])
    report["prior_cfg"] = asdict(args.prior)
    with open(osp.join(args.out, "structure.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- console summary
    W, K, R, Bn = report["weights"], report["knn"], report["rigidity"], report["bones"]
    print(f"\n== {args.ckpt} (epoch {report['epoch']})")
    mw = W["mean_max_weight_opaque"]
    mw_s = "n/a" if mw is None else f"{mw:.3f}"
    print(f"fg={W['num_fg']} opaque={W['num_opaque']}  mean max w (opaque)="
          f"{mw_s}  entropy={W['mean_entropy']:.2f}/"
          f"{W['uniform_entropy']:.2f}  dominant bases={W['bases_dominant_gt_1000']}")
    print(f"argmax hist {W['argmax_hist']}")
    print(f"kNN: E={K['num_edges']} TV={K['tv_charbonnier']:.4f} median L1 dw="
          f"{K['median_l1_weight_diff']:.3f} cut@tau={K['frac_edges_cut_at_tau']:.1%}")
    for r in R:
        print(f"rigidity (kNN @ frame {r['frame']}): median std/mean="
              f"{r['median_std_over_mean']:.3f}  >20%: {r['frac_gt_20pct']:.1%}  "
              f">5%: {r['frac_gt_5pct']:.1%}")
    print(f"bones: M={Bn['num_bones']} valid={Bn['num_valid']} largest="
          f"{Bn['largest_sizes'][:10]}")
    print(f"joints: {report['joint_class_counts']}")
    for j in sorted(joints, key=lambda j: -j["contact_count"])[:15]:
        ms = f" merge={j['merge_score']:.3f}" if "merge_score" in j else ""
        print(f"  {j['a']:>4} - {str(j['b']):>6}  n={j['contact_count']:>5} "
              f"{j['class']:>5} res={j['res_norm']:.2e} "
              f"eig={[round(x, 4) for x in j['eig_H0_over_T']]}{ms}")
    print(f"wrote {args.out}/structure.json and bones_canonical.ply")


if __name__ == "__main__":
    main(tyro.cli(Args))
