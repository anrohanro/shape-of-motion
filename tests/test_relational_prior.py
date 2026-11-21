"""Unit tests for flow3d/relational_prior.py.

Run:  CUDA_VISIBLE_DEVICES=1 python tests/test_relational_prior.py   (pytest-compatible)
Synthetic cases with known answers run on CPU (float64 where precision
matters). Only test_checkpoint_back_compat uses CUDA; pin it to an idle GPU
with CUDA_VISIBLE_DEVICES so it never lands on a GPU running training.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import roma
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow3d.configs import PriorConfig  # noqa: E402
from flow3d.params import GaussianParams, MotionBases  # noqa: E402
from flow3d.relational_prior import (  # noqa: E402
    JOINT_BALL,
    JOINT_HINGE,
    JOINT_RIGID,
    STATIC,
    RelationalPrior,
    bone_mean_weights,
    build_knn_edges,
    charbonnier_tv,
    classify_joint,
    extract_bones,
    merge_score,
    solve_joint,
)
from flow3d.scene_model import SceneModel  # noqa: E402
from flow3d.transforms import rmat_to_cont_6d  # noqa: E402

DT = torch.float64
BETA_REL = PriorConfig().beta_rel  # production regulariser


class Skip(Exception):
    pass


def _skip(msg):
    try:
        import pytest

        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


def _rand_rot(n, scale=1.0, gen=None):
    v = torch.randn(n, 3, generator=gen, dtype=DT) * scale
    return roma.rotvec_to_rotmat(v)


def _axis_rot(axis, angles):
    axis = axis / axis.norm()
    return roma.rotvec_to_rotmat(angles[:, None] * axis[None])


def _apply(R, t, x):
    """R (T,3,3), t (T,3), x (n,3) -> (n,T,3)"""
    return torch.einsum("tij,nj->nti", R, x) + t[None]


def _pair_from_relative(T, Q, offset, gen):
    """Bone a random; bone b = a composed with relative (Q_t, offset_t)."""
    Ra = _rand_rot(T, 0.8, gen)
    ta = torch.randn(T, 3, generator=gen, dtype=DT)
    Rb = Ra @ Q
    tb = torch.einsum("tij,tj->ti", Ra, offset) + ta
    return Ra, ta, Rb, tb


# --------------------------------------------------------------------------- #
def test_tv_zero_iff_equal_and_grad_flows():
    logits = torch.randn(1, 6, dtype=DT).repeat(5, 1).requires_grad_(True)
    w = torch.softmax(logits, -1)
    edges = torch.tensor([[0, 1], [1, 2], [3, 4]])
    assert charbonnier_tv(w, edges).abs().item() < 1e-12
    logits2 = torch.randn(5, 6, dtype=DT, requires_grad=True)
    tv = charbonnier_tv(torch.softmax(logits2, -1), edges)
    assert tv.item() > 0
    tv.backward()
    assert torch.isfinite(logits2.grad).all() and logits2.grad.abs().sum() > 0
    assert charbonnier_tv(w, torch.zeros(0, 2, dtype=torch.long)).item() == 0


def test_ball_joint_recovered():
    gen = torch.Generator().manual_seed(0)
    T = 40
    p = torch.tensor([0.3, -0.2, 0.5], dtype=DT)
    Q = _rand_rot(T, 0.7, gen)
    offset = p[None] - torch.einsum("tij,j->ti", Q, p)  # rotation about p
    Ra, ta, Rb, tb = _pair_from_relative(T, Q, offset, gen)
    c = p + torch.tensor([0.3, -0.4, 0.2], dtype=DT)  # prior far from the pivot
    sol = solve_joint(Ra[None], ta[None], Rb[None], tb[None], c[None], BETA_REL * T)
    assert (sol.p[0] - p).norm() < 1e-6, sol.p  # ridge must not bias p
    assert sol.res[0] < 1e-10
    assert sol.evals[0, 0] > 1e-2  # not a hinge
    cls = classify_joint(sol.evals, sol.res, 0.0076, 4e-4, 0.05)
    assert cls[0] == JOINT_BALL


def test_hinge_axis_and_point():
    gen = torch.Generator().manual_seed(1)
    T = 40
    p = torch.tensor([0.1, 0.4, -0.3], dtype=DT)
    n = torch.tensor([1.0, 2.0, -0.5], dtype=DT)
    n = n / n.norm()
    Q = _axis_rot(n, torch.linspace(-1.2, 1.5, T, dtype=DT))
    offset = p[None] - torch.einsum("tij,j->ti", Q, p)
    Ra, ta, Rb, tb = _pair_from_relative(T, Q, offset, gen)
    # c offset perpendicular to the axis by 0.3 (worst case for a ridge spring)
    perp = torch.linalg.cross(n, torch.tensor([0.0, 0.0, 1.0], dtype=DT))
    c = p + 0.3 * perp / perp.norm() + 0.2 * n
    sol = solve_joint(Ra[None], ta[None], Rb[None], tb[None], c[None], BETA_REL * T)
    l = sol.evals[0]
    assert l[0] < 1e-10 and l[1] > 1e-2, l
    assert abs(float(sol.n1[0] @ n)) > 0.999
    expect = p + ((c - p) @ n) * n  # point on the axis nearest c
    assert (sol.p[0] - expect).norm() < 1e-6, (sol.p, expect)
    assert sol.res[0] < 1e-12
    cls = classify_joint(sol.evals, sol.res, 0.0076, 4e-4, 0.05)
    assert cls[0] == JOINT_HINGE


def test_exact_joint_has_zero_gradient():
    """Exact hinge with c far from the axle: no spring force toward c."""
    gen = torch.Generator().manual_seed(11)
    T = 40
    p = torch.tensor([0.1, 0.4, -0.3], dtype=DT)
    n = torch.tensor([0.0, 0.6, 0.8], dtype=DT)
    Q = _axis_rot(n, torch.linspace(-1.0, 1.4, T, dtype=DT))
    offset = p[None] - torch.einsum("tij,j->ti", Q, p)
    Ra, ta, Rb, tb = _pair_from_relative(T, Q, offset, gen)
    ta = ta.clone().requires_grad_(True)
    c = p + torch.tensor([0.3, 0.0, 0.0], dtype=DT)
    sol = solve_joint(Ra[None], ta[None], Rb[None], tb[None], c[None], BETA_REL * T)
    sol.res.sum().backward()
    assert ta.grad.abs().max() < 1e-9, ta.grad.abs().max()


def test_static_pair_mean_anchor_no_privileged_frame():
    """Static formulation (partner = temporal mean) recovers a world-fixed
    pivot and spreads gradient evenly over frames."""
    gen = torch.Generator().manual_seed(12)
    T = 30
    p = torch.tensor([0.2, -0.1, 0.3], dtype=DT)  # canonical pivot
    q = torch.tensor([1.0, 0.5, -0.2], dtype=DT)  # fixed world location
    R = _rand_rot(T, 0.8, gen).requires_grad_(True)
    t = (q[None] - torch.einsum("tij,j->ti", R, p)).detach().requires_grad_(True)
    Rs = R.mean(0, keepdim=True).expand_as(R)
    ts_ = t.mean(0, keepdim=True).expand_as(t)
    c = torch.zeros(3, dtype=DT)
    sol = solve_joint(R[None], t[None], Rs[None], ts_[None], c[None], BETA_REL * T)
    assert (sol.p[0] - p).norm() < 1e-6 and sol.res[0] < 1e-12
    # perturb and check no single frame dominates the gradient
    t2 = (t.detach() + 0.01 * torch.randn(T, 3, generator=gen, dtype=DT)).requires_grad_(True)
    sol2 = solve_joint(R.detach()[None], t2[None],
                       R.detach().mean(0, keepdim=True).expand_as(R)[None],
                       t2.mean(0, keepdim=True).expand_as(t2)[None], c[None], BETA_REL * T)
    sol2.res.sum().backward()
    gn = t2.grad.norm(dim=-1)
    assert gn.max() < 5 * gn.median(), gn


def test_no_static_contact_with_default_config():
    """Default contact radius + bg far away (like paper-windmill): no static
    pairs, no crash, fg-fg joint still present."""
    model = _toy_scene()
    with torch.no_grad():
        model.bg.params["means"] += torch.tensor([0.0, 0.0, -0.6])  # move stick away
    cfg = PriorConfig(w_joint=0.1, joint_start_epoch=0, bone_n_min=50, adj_min_edges=5)
    prior = RelationalPrior(cfg, model.num_frames)
    loss, stats = prior.loss(model, epoch=0, global_step=10**6)
    assert torch.isfinite(loss)
    pairs = prior.bones.pairs.tolist()
    assert all(p[1] != STATIC for p in pairs) and len(pairs) >= 1, pairs


def test_periodic_knn_rebuild_does_not_force_bone_rebuild():
    model = _toy_scene(with_bg=False)
    cfg = PriorConfig(w_joint=0.1, joint_start_epoch=0, knn_every_epochs=2,
                      bone_every_epochs=5, bone_n_min=50, adj_min_edges=5)
    prior = RelationalPrior(cfg, model.num_frames)
    rebuilt = []
    for ep in range(11):
        before = prior.last_bone_epoch
        prior.loss(model, epoch=ep, global_step=10**6)
        if prior.last_bone_epoch != before:
            rebuilt.append(ep)
    assert rebuilt == [0, 5, 10], rebuilt


def test_rigid_weld_and_merge_score():
    gen = torch.Generator().manual_seed(2)
    T = 30
    Ra = _rand_rot(T, 0.8, gen)
    ta = torch.randn(T, 3, generator=gen, dtype=DT)
    # Same transform -> same rigid body: all eigenvalues 0, residual 0.
    sol = solve_joint(Ra[None], ta[None], Ra[None], ta[None], torch.zeros(1, 3, dtype=DT), 1e-2 * T)
    assert sol.evals.abs().max() < 1e-12 and sol.res.abs().max() < 1e-12
    cls = classify_joint(sol.evals, sol.res, 0.0076, 4e-4, 0.05)
    assert cls[0] == JOINT_RIGID
    # Merge test: b = a composed with a constant offset G -> rigid with a.
    G_R = _rand_rot(1, 0.5, gen)[0]
    G_t = torch.randn(3, generator=gen, dtype=DT)
    mu_b = torch.randn(50, 3, generator=gen, dtype=DT) * 0.2
    Rb, tb = Ra @ G_R, torch.einsum("tij,j->ti", Ra, G_t) + ta
    traj_b = _apply(Rb, tb, mu_b)
    diam = 0.8
    assert merge_score(Ra, ta, traj_b, diam) < 1e-9
    # Hinged b -> clearly not mergeable.
    Q = _axis_rot(torch.tensor([0.0, 0.0, 1.0], dtype=DT), torch.linspace(0, 1.5, T, dtype=DT))
    traj_h = _apply(Ra @ Q, ta, mu_b + torch.tensor([0.5, 0, 0], dtype=DT))
    assert merge_score(Ra, ta, traj_h, diam) > 0.05


def test_envelope_gradient_matches_finite_difference():
    gen = torch.Generator().manual_seed(3)
    T = 25
    p = torch.tensor([0.2, 0.1, -0.4], dtype=DT)
    Q = _rand_rot(T, 0.6, gen)
    offset = p[None] - torch.einsum("tij,j->ti", Q, p)
    Ra, ta, Rb, tb0 = _pair_from_relative(T, Q, offset, gen)
    d = torch.randn(T, 3, generator=gen, dtype=DT)  # perturbation direction
    c = torch.zeros(3, dtype=DT)
    beta = BETA_REL * T

    def fmin(theta):
        # all directions determined (random relative rotations) -> p* is the
        # exact minimiser of f, so the minimised objective / T is res
        tb = tb0 + theta * d
        sol = solve_joint(Ra[None], ta[None], Rb[None], tb[None], c[None], beta)
        assert (sol.evals[0] * T > beta).all()
        return sol.res[0]

    theta0 = 0.3
    th = torch.tensor(theta0, dtype=DT, requires_grad=True)
    sol = solve_joint(Ra[None], ta[None], Rb[None], (tb0 + th * d)[None], c[None], beta)
    sol.res[0].backward()
    h = 1e-6
    fd = (fmin(theta0 + h) - fmin(theta0 - h)) / (2 * h)
    assert abs(th.grad.item() - fd.item()) < 1e-6 * max(1.0, abs(fd.item())), (
        th.grad.item(),
        fd.item(),
    )


def test_extract_bones_blocks_and_absorb():
    # 3 blocks along x, 300 points each; weights constant per block.
    gen = torch.Generator().manual_seed(4)
    xs, ws = [], []
    block_w = torch.softmax(torch.randn(3, 5, generator=gen) * 3, -1)
    for b in range(3):
        g = torch.stack(
            torch.meshgrid(
                torch.arange(10) + 10 * b, torch.arange(10), torch.arange(3), indexing="ij"
            ),
            -1,
        ).reshape(-1, 3)
        xs.append(g.float() * 0.1)
        ws.append(block_w[b].expand(g.shape[0], -1))
    x = torch.cat(xs)
    w = torch.cat(ws).clone()
    # Noisy weights on a few points of block 1 -> tiny components to absorb.
    # Interior of block 1 (index = 300 + ix*30 + iy*3 + iz, ix=iy=5), so the
    # only sensible absorber is block 1 itself.
    noisy = 300 + 5 * 30 + 5 * 3 + torch.tensor([0, 1, 2, 3, 4])
    w[noisy] = torch.softmax(torch.randn(5, 5, generator=gen) * 5, -1)
    edges, _ = build_knn_edges(x, k=6)
    bones = extract_bones(w, edges, x, tau=0.05, n_min=50, m_adj=5)
    assert bones.num_bones == 3, bones.sizes
    assert sorted(bones.sizes.tolist()) == [300, 300, 300]
    pairs = {tuple(sorted(p)) for p in bones.pairs.tolist()}
    lab = [int(bones.labels[i]) for i in (0, 450, 899)]  # one point per block
    expect = {tuple(sorted((lab[0], lab[1]))), tuple(sorted((lab[1], lab[2])))}
    assert pairs == expect, (pairs, expect)
    # Centroid of the 0-1 boundary lies near x = 0.95.
    k = [i for i, p in enumerate(bones.pairs.tolist()) if set(p) == {lab[0], lab[1]}][0]
    assert abs(bones.centers[k, 0].item() - 0.95) < 0.06


def test_knn_handles_duplicates():
    x = torch.rand(100, 3)
    x = torch.cat([x, x[:10]])  # exact duplicates, as densify "dup" creates
    edges, med = build_knn_edges(x, k=4, max_dist_factor=0)
    assert (edges[:, 0] < edges[:, 1]).all()
    assert edges.max() < x.shape[0]
    # every node has at least one edge
    deg = torch.bincount(edges.reshape(-1), minlength=x.shape[0])
    assert (deg > 0).all()


def test_hinge_grad_finite_for_rigid_pair():
    T = 10
    R6 = torch.randn(T, 6, dtype=DT, requires_grad=True)
    from flow3d.transforms import cont_6d_to_rmat

    Ra = cont_6d_to_rmat(R6)
    ta = torch.zeros(T, 3, dtype=DT)
    sol = solve_joint(Ra[None], ta[None], Ra.detach()[None], ta[None], torch.zeros(1, 3, dtype=DT), 0.1)
    (sol.hinge.sum() + sol.res.sum()).backward()
    assert torch.isfinite(R6.grad).all()


# --------------------------------------------------------------------------- #
def _toy_scene(n_per=400, T=20, seed=5, with_bg=True):
    """Two touching blocks driven by two different bases; block 0 spins
    about a fixed world axis (hinge to the static world)."""
    g = torch.Generator().manual_seed(seed)
    blocks = []
    for b in range(2):
        pts = torch.rand(n_per, 3, generator=g) * torch.tensor([0.5, 0.5, 0.2])
        pts[:, 0] += 0.5 * b
        blocks.append(pts)
    means = torch.cat(blocks)
    n = means.shape[0]
    B = 3
    logits = torch.full((n, B), -8.0)
    logits[:n_per, 0] = 8.0
    logits[n_per:, 1] = 8.0
    angles = torch.linspace(0, 1.0, T)
    R0 = roma.rotvec_to_rotmat(angles[:, None] * torch.tensor([[0.0, 0.0, 1.0]]))
    R1 = roma.rotvec_to_rotmat(torch.randn(T, 3, generator=g) * 0.3)
    R2 = torch.eye(3).expand(T, 3, 3)
    rots = torch.stack([rmat_to_cont_6d(R) for R in (R0, R1, R2)])  # (B, T, 6)
    transls = torch.zeros(B, T, 3)
    transls[1] = torch.randn(T, 3, generator=g) * 0.1
    fg = GaussianParams(
        means,
        torch.randn(n, 4, generator=g),
        torch.full((n, 3), -4.0),
        torch.zeros(n, 3),
        torch.zeros(n),
        motion_coefs=logits,
    )
    mb = MotionBases(rots, transls)
    bg = None
    if with_bg:
        # Static background "stick" touching block 0 around the spin axis.
        bgm = torch.rand(300, 3, generator=g) * torch.tensor([0.06, 0.06, 0.6])
        bgm[:, 2] -= 0.62
        bg = GaussianParams(
            bgm, torch.randn(300, 4, generator=g), torch.full((300, 3), -4.0),
            torch.zeros(300, 3), torch.zeros(300),
        )
    Ks = torch.eye(3)[None].repeat(T, 1, 1)
    w2cs = torch.eye(4)[None].repeat(T, 1, 1)
    return SceneModel(Ks, w2cs, fg, mb, None, bg)


def test_relational_prior_end_to_end_and_stale_guard():
    model = _toy_scene()
    cfg = PriorConfig(
        w_tv=0.1, w_joint=0.1, w_hinge=0.01, joint_start_epoch=0,
        bone_n_min=50, adj_min_edges=5, contact_dist_factor=3.0,
    )
    prior = RelationalPrior(cfg, model.num_frames)
    loss, stats = prior.loss(model, epoch=0, global_step=10**6)
    assert torch.isfinite(loss) and loss.item() > 0, stats
    assert stats["train/num_valid_bones"] == 2, stats
    pairs = prior.bones.pairs.tolist()
    assert any(p[1] != STATIC for p in pairs), pairs  # block-block joint
    assert any(p[1] == STATIC for p in pairs), pairs  # block 0 <-> static stick
    loss.backward()
    for name in ("motion_coefs",):
        gr = model.fg.params[name].grad
        assert gr is not None and torch.isfinite(gr).all()
    gr = model.motion_bases.params["rots"].grad
    assert gr is not None and torch.isfinite(gr).all() and gr.abs().sum() > 0

    # Static joint of the spinning block should be a hinge about z.
    sol, res_n, cls = prior.joint_terms(model)
    k = [i for i, p in enumerate(pairs) if p[1] == STATIC][0]
    assert cls[k] == JOINT_HINGE, (cls, sol.evals[k], res_n[k])
    assert abs(sol.n1[k, 2].item()) > 0.99

    # Stale-index guard: simulate a densify that grows N without mark_dirty.
    old_e = prior.edges.shape[0]
    model.fg.params["means"] = torch.nn.Parameter(
        torch.cat([model.fg.params["means"].data] * 2)
    )
    for k2 in ("quats", "scales", "colors", "opacities", "motion_coefs"):
        model.fg.params[k2] = torch.nn.Parameter(torch.cat([model.fg.params[k2].data] * 2))
    prior.mark_dirty()
    prior.loss(model, epoch=1, global_step=10**6)
    assert prior.edges.max() < model.num_fg_gaussians
    assert prior.bones.labels.shape[0] == model.num_fg_gaussians
    assert prior.edges.shape[0] != old_e
    # Shrink without mark_dirty: guard must trigger a rebuild.
    n_half = model.num_fg_gaussians // 2
    for k2 in ("means", "quats", "scales", "colors", "opacities", "motion_coefs"):
        model.fg.params[k2] = torch.nn.Parameter(model.fg.params[k2].data[:n_half])
    prior.loss(model, epoch=1, global_step=10**6)
    assert prior.edges.max() < model.num_fg_gaussians
    assert prior.bones.labels.shape[0] == model.num_fg_gaussians


def test_prior_disabled_is_noop():
    model = _toy_scene(with_bg=False)
    prior = RelationalPrior(PriorConfig(), model.num_frames)
    loss, stats = prior.loss(model, epoch=500)
    assert loss.item() == 0 and stats == {} and prior.edges is None


def test_bone_mean_weights():
    w = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    lab = torch.tensor([0, 0, 1])
    wb = bone_mean_weights(w, lab, 2)
    assert torch.allclose(wb, torch.tensor([[0.5, 0.5], [0.5, 0.5]]))


def test_checkpoint_back_compat():
    ckpt = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs/paper-windmill/checkpoints/last.ckpt",
    )
    if not os.path.exists(ckpt) or not torch.cuda.is_available():
        _skip("no baseline checkpoint / no CUDA")
    from flow3d.configs import LossesConfig, OptimizerConfig, SceneLRConfig
    from flow3d.configs import (
        BGLRConfig, CameraPoseLRConfig, CameraScalesLRConfig, FGLRConfig, MotionLRConfig,
    )
    from flow3d.trainer import Trainer

    lr = SceneLRConfig(
        FGLRConfig(), BGLRConfig(), MotionLRConfig(), CameraPoseLRConfig(), CameraScalesLRConfig()
    )
    with tempfile.TemporaryDirectory() as d:
        trainer, epoch = Trainer.init_from_checkpoint(
            ckpt, torch.device("cuda"), False, lr, LossesConfig(), OptimizerConfig(),
            work_dir=d, prior_cfg=None,
        )
        assert trainer.prior is None and epoch > 0
        trainer2, _ = Trainer.init_from_checkpoint(
            ckpt, torch.device("cuda"), False, lr, LossesConfig(), OptimizerConfig(),
            work_dir=d, prior_cfg=PriorConfig(w_tv=0.1),
        )
        assert trainer2.prior is not None
        loss, stats = trainer2.prior.loss(trainer2.model, epoch=epoch)
        assert torch.isfinite(loss) and stats["train/tv_loss"] > 0
        print(f"  baseline ckpt tv_loss={stats['train/tv_loss']:.4f} "
              f"edges={int(stats['train/prior_num_edges'])}")
        del trainer, trainer2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Skip as e:
            skipped += 1
            print(f"SKIP {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback

            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed, {skipped} skipped")
    sys.exit(1 if failed else 0)
