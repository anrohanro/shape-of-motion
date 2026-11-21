"""Relational motion priors for Shape of Motion.

Level 1: Charbonnier (smoothed L1) total variation on the motion weights of
         neighbouring Gaussians (kNN graph on canonical means).
Level 2: bones = connected components of near-equal weights; a closed-form
         joint residual between adjacent bones, plus a hinge term.

Nothing here owns an ``nn.Parameter``: every prior is a pure function of the
existing scene parameters, so optimizers and checkpoints are unaffected. The
stateful ``RelationalPrior`` only caches graph structure (kNN edges, bone
labels), which is recomputed whenever Gaussians are densified / culled.

See docs: "Relational Motion Priors for Shape of Motion", sections 4-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from torch import Tensor

# Joint classes (doc section 6 table).
JOINT_RIGID, JOINT_SLIDE, JOINT_HINGE, JOINT_BALL, JOINT_NONE = range(5)
JOINT_NAMES = ["rigid", "slide", "hinge", "ball", "none"]
STATIC = -1  # partner index used for "bone touches the static background"


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_knn_edges(
    means: Tensor, k: int, max_dist_factor: float = 3.0
) -> tuple[Tensor, float]:
    """Undirected kNN graph on points.

    Returns
        edges: (E, 2) int64 on means.device, unique, i < j.
        med:   median kNN distance (float, scene units).
    Exact duplicate points (created by densify "dup") are handled: self-pairs
    are removed by index, not by dropping the first neighbour column.
    """
    device = means.device
    x = means.detach().float().cpu().numpy()
    n = x.shape[0]
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return torch.zeros(0, 2, dtype=torch.long, device=device), 0.0
    tree = cKDTree(x)
    d, idx = tree.query(x, k=k_eff + 1)
    rows = np.repeat(np.arange(n), k_eff + 1)
    d, idx = d.reshape(-1), idx.reshape(-1)
    keep = idx != rows
    rows, idx, d = rows[keep], idx[keep], d[keep]
    med = float(np.median(d)) if d.size else 0.0
    if med > 0 and max_dist_factor > 0:
        keep = d <= max_dist_factor * med
        rows, idx = rows[keep], idx[keep]
    a = np.minimum(rows, idx).astype(np.int64)
    b = np.maximum(rows, idx).astype(np.int64)
    key = np.unique(a * n + b)
    edges = np.stack([key // n, key % n], axis=1)
    return torch.from_numpy(edges).to(device), med


def robust_extent(means: Tensor, q: float = 0.9) -> float:
    """Length scale of a point set: q-quantile distance to the median point."""
    m = means.detach().float()
    if m.shape[0] == 0:
        return 1.0
    c = m.median(dim=0).values
    r = torch.quantile((m - c).norm(dim=-1), q).item()
    return max(r, 1e-6)


# --------------------------------------------------------------------------- #
# Level 1
# --------------------------------------------------------------------------- #
def charbonnier_tv(w: Tensor, edges: Tensor, eps: float = 1e-3) -> Tensor:
    """mean_e sum_b (sqrt((w_ib - w_jb)^2 + eps^2) - eps); exactly 0 iff equal."""
    if edges.numel() == 0:
        return w.sum() * 0.0
    diff = w[edges[:, 0]] - w[edges[:, 1]]
    return (torch.sqrt(diff * diff + eps * eps) - eps).sum(-1).mean()


# --------------------------------------------------------------------------- #
# Bones
# --------------------------------------------------------------------------- #
@dataclass
class Bones:
    labels: Tensor  # (N,) int64, bone id per Gaussian
    num_bones: int
    sizes: Tensor  # (M,) int64
    valid: Tensor  # (M,) bool, size >= n_min (eligible for joints)
    boundary: Tensor  # (N,) bool, has at least one cut kNN edge
    pairs: Tensor  # (P, 2) int64, (a, b); b == STATIC for background
    pair_counts: Tensor  # (P,) int64, crossing kNN edges / contact Gaussians
    centers: Tensor  # (P, 3) canonical boundary centroid c_ab


def _absorb_small(
    labels: np.ndarray, ei: np.ndarray, ej: np.ndarray, n_min: int, max_iter: int = 20
) -> np.ndarray:
    """Merge components smaller than n_min into the neighbour sharing most edges.

    All small components are resolved simultaneously: each small component
    points at its best neighbour, and components of that "merge graph" are
    fused. Every small node has exactly one outgoing edge and large nodes none,
    so a fused group contains at most one large component (no large-large
    merges), and mutual small-small choices (cycles) are fused, not swapped.
    """
    for _ in range(max_iter):
        _, labels = np.unique(labels, return_inverse=True)
        c = labels.max() + 1
        sizes = np.bincount(labels, minlength=c)
        small = sizes < n_min
        if not small.any():
            break
        la, lb = labels[ei], labels[ej]
        cross = la != lb
        la, lb = la[cross], lb[cross]
        # Both directions: a small component may sit on either end of an edge.
        src = np.concatenate([la, lb])
        dst = np.concatenate([lb, la])
        sel = small[src]
        src, dst = src[sel], dst[sel]
        if src.size == 0:
            break  # remaining small components are isolated
        key, cnt = np.unique(src * c + dst, return_counts=True)
        s, d = key // c, key % c
        order = np.lexsort((-cnt, s))  # by src, then count descending
        s, d = s[order], d[order]
        first = np.ones_like(s, dtype=bool)
        first[1:] = s[1:] != s[:-1]
        s, d = s[first], d[first]
        g = coo_matrix((np.ones_like(s), (s, d)), shape=(c, c))
        _, comp = connected_components(g, directed=False)
        labels = comp[labels]
    _, labels = np.unique(labels, return_inverse=True)
    return labels


@torch.no_grad()
def extract_bones(
    w: Tensor,
    edges: Tensor,
    means: Tensor,
    tau: float,
    n_min: int,
    m_adj: int,
) -> Bones:
    """Bones = connected components of kNN edges with ||w_i - w_j||_1 <= tau."""
    device = w.device
    n = w.shape[0]
    e = edges.cpu().numpy()
    ei, ej = e[:, 0], e[:, 1]
    l1 = (w[edges[:, 0]] - w[edges[:, 1]]).abs().sum(-1).cpu().numpy()
    keep = l1 <= tau
    g = coo_matrix((np.ones(keep.sum()), (ei[keep], ej[keep])), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    labels = _absorb_small(labels, ei, ej, n_min)
    m = int(labels.max()) + 1 if n > 0 else 0
    sizes = np.bincount(labels, minlength=m)
    valid = sizes >= n_min

    la, lb = labels[ei], labels[ej]
    cut = la != lb
    boundary = np.zeros(n, dtype=bool)
    boundary[ei[cut]] = True
    boundary[ej[cut]] = True

    # Adjacency between valid bones with >= m_adj crossing edges.
    sel = cut & valid[la] & valid[lb]
    a = np.minimum(la[sel], lb[sel])
    b = np.maximum(la[sel], lb[sel])
    key, inv, cnt = np.unique(a * m + b, return_inverse=True, return_counts=True)
    ok = cnt >= m_adj
    pa, pb = key[ok] // max(m, 1), key[ok] % max(m, 1)
    # Boundary centroid c_ab: mean canonical position of crossing-edge endpoints.
    mu = means.detach().float().cpu().numpy()
    mid = 0.5 * (mu[ei[sel]] + mu[ej[sel]])
    sums = np.zeros((key.size, 3))
    np.add.at(sums, inv, mid)
    centers = (sums / cnt[:, None])[ok]

    return Bones(
        labels=torch.from_numpy(labels.astype(np.int64)).to(device),
        num_bones=m,
        sizes=torch.from_numpy(sizes.astype(np.int64)).to(device),
        valid=torch.from_numpy(valid).to(device),
        boundary=torch.from_numpy(boundary).to(device),
        pairs=torch.from_numpy(np.stack([pa, pb], 1).astype(np.int64)).to(device),
        pair_counts=torch.from_numpy(cnt[ok].astype(np.int64)).to(device),
        centers=torch.from_numpy(centers.astype(np.float32)).to(device),
    )


@torch.no_grad()
def static_contacts(
    bones: Bones,
    traj: Tensor,
    means: Tensor,
    bg_tree: cKDTree,
    contact_dist: float,
    m_adj: int,
    min_frac: float = 0.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Bones persistently touching the static background.

    traj: (N, S, 3) world positions of the fg Gaussians at S sampled frames.
    A bone touches the background at a frame if >= m_adj of its Gaussians are
    within contact_dist of a background Gaussian; it is a static-contact bone
    if that holds on >= min_frac of the sampled frames.
    Returns (bone ids (Q,), contact counts (Q,), canonical centroids (Q, 3)).
    """
    device = traj.device
    n, s, _ = traj.shape
    pts = traj.reshape(-1, 3).float().cpu().numpy()
    d, _ = bg_tree.query(pts, k=1, distance_upper_bound=contact_dist)
    touch = torch.from_numpy(np.isfinite(d).reshape(n, s)).to(device)  # (N, S)
    m = bones.num_bones
    per_frame = torch.zeros(m, s, dtype=torch.long, device=device)
    per_frame.index_add_(0, bones.labels, touch.long())
    frac = (per_frame >= m_adj).float().mean(dim=1)
    is_contact = (frac >= min_frac) & bones.valid
    ids = torch.nonzero(is_contact).squeeze(-1)
    touch_any = touch.float().mean(dim=1) >= min_frac  # (N,)
    counts, cents = [], []
    for a in ids.tolist():
        sel = (bones.labels == a) & touch_any
        if sel.sum() == 0:
            sel = bones.labels == a
        counts.append(int(sel.sum()))
        cents.append(means[sel].float().mean(0))
    if len(cents) == 0:
        return ids, torch.zeros(0, dtype=torch.long, device=device), torch.zeros(
            0, 3, device=device
        )
    return ids, torch.tensor(counts, device=device), torch.stack(cents)


def bone_mean_weights(w: Tensor, labels: Tensor, num_bones: int) -> Tensor:
    """wbar_a = mean of member weights (differentiable)."""
    sums = torch.zeros(num_bones, w.shape[1], device=w.device, dtype=w.dtype)
    sums = sums.index_add(0, labels, w)
    cnt = torch.bincount(labels, minlength=num_bones).clamp_min(1).to(w.dtype)
    return sums / cnt[:, None]


# --------------------------------------------------------------------------- #
# Level 2: joints
# --------------------------------------------------------------------------- #
@dataclass
class JointSolution:
    res: Tensor  # (P,) mean_t ||M_t p* + u_t||^2 (differentiable, float32)
    p: Tensor  # (P, 3) joint position (canonical), detached
    evals: Tensor  # (P, 3) ascending eigenvalues of H0 / T, detached
    n1: Tensor  # (P, 3) eigenvector of smallest eigenvalue, detached
    hinge: Tensor  # (P,) Rayleigh quotient n1^T (H0/T) n1 (differentiable)


def solve_joint(
    Ra: Tensor, ta: Tensor, Rb: Tensor, tb: Tensor, c: Tensor, beta: float
) -> JointSolution:
    """Closed-form joint between bone pairs (doc Prop. 6.1 / 6.2).

    Ra, Rb: (P, T, 3, 3); ta, tb: (P, T, 3); c: (P, 3) prior joint location.
    Minimises f(p) = sum_t ||M_t p + u_t||^2 with M_t = Ra - Rb, u_t = ta - tb.
    In the eigenbasis H0 = V diag(l) V^T (H0 = sum_t M_t^T M_t, g = sum M^T u):
      * determined directions (l_k > beta): exact least squares z_k = -(V^T g)_k / l_k
      * near-null directions (l_k <= beta, e.g. the hinge axis, or all three
        for a rigid pair): ridge toward c, z_k = (-(V^T g)_k + beta (V^T c)_k) / (l_k + beta)
    The ridge acts only where f does not determine p, so grad_p f = 0 on the
    determined subspace and the envelope gradient (p* held constant in the
    backward pass) carries no spurious spring force toward c. Done in
    float64; p* and the eigenvector are detached.
    """
    T = Ra.shape[1]
    M = Ra - Rb  # (P, T, 3, 3)
    u = ta - tb  # (P, T, 3)
    with torch.no_grad():
        Md, ud = M.double(), u.double()
        H0 = torch.einsum("ptji,ptjk->pik", Md, Md)
        g = torch.einsum("ptji,ptj->pi", Md, ud)
        lam, V = torch.linalg.eigh(H0)  # ascending
        b = -torch.einsum("pji,pj->pi", V, g)  # V^T (-g)
        cc = torch.einsum("pji,pj->pi", V, c.double())  # V^T c
        det = lam > beta
        z = torch.where(
            det, b / lam.clamp_min(1e-300), (b + beta * cc) / (lam + beta)
        )
        p = torch.einsum("pij,pj->pi", V, z)
        evals, n1 = lam / T, V[..., 0]
        p, evals, n1 = p.to(M.dtype), evals.to(M.dtype), n1.to(M.dtype)
    r = torch.einsum("ptij,pj->pti", M, p) + u
    res = (r * r).sum(-1).mean(-1)
    Mn = torch.einsum("ptij,pj->pti", M, n1)
    hinge = (Mn * Mn).sum(-1).mean(-1)
    return JointSolution(res=res, p=p, evals=evals, n1=n1, hinge=hinge)


def geman_mcclure(x: Tensor, c: float) -> Tensor:
    return x / (x + c * c)


@torch.no_grad()
def classify_joint(
    evals: Tensor, res_n: Tensor, rot_eps: float, res_eps: float, hinge_ratio: float
) -> Tensor:
    """Doc section 6 table. evals: (P,3) ascending eigenvalues of H0/T
    (unitless); res_n: (P,) residual normalised by scale^2."""
    l1, l2, l3 = evals[:, 0], evals[:, 1], evals[:, 2]
    small_res = res_n < res_eps
    no_rot = l3 < rot_eps
    out = torch.full_like(l1, JOINT_NONE, dtype=torch.long)
    out[~small_res & no_rot] = JOINT_SLIDE
    ball = small_res & ~no_rot
    out[ball] = JOINT_BALL
    hinge = ball & (l1 < hinge_ratio * l2)
    out[hinge] = JOINT_HINGE
    out[small_res & no_rot] = JOINT_RIGID
    return out


@torch.no_grad()
def merge_score(Ra: Tensor, ta: Tensor, traj_b: Tensor, diam_b: float) -> float:
    """Doc Prop. 5.1: b's points seen from a's frame should be constant.

    Ra (T,3,3), ta (T,3), traj_b (n,T,3) world trajectories.
    Score = max_t median_i ||xt_i(t) - median_t xt_i|| / diam_b
    (the temporal median is used as reference instead of one frame t0, for
    robustness). < ~0.01-0.02 means b is rigidly attached to a.
    """
    xt = torch.einsum("tji,ntj->nti", Ra, traj_b - ta[None])  # R^T (x - t)
    ref = xt.median(dim=1, keepdim=True).values
    dev = (xt - ref).norm(dim=-1)  # (n, T)
    return float(dev.median(dim=0).values.max() / max(diam_b, 1e-9))


# --------------------------------------------------------------------------- #
# Stateful wrapper used by the trainer
# --------------------------------------------------------------------------- #
class RelationalPrior:
    def __init__(self, cfg, num_frames: int):
        self.cfg = cfg
        self.num_frames = num_frames
        self.edges: Tensor | None = None
        self.knn_med = 0.0
        self.fg_scale = 1.0
        self.last_knn_epoch = -(10**9)
        self.bones: Bones | None = None
        self.last_bone_epoch = -(10**9)
        self.dirty = True
        self.bones_dirty = True
        self._warned_no_contact = False
        # Set by the trainer: densify/cull stop at this global step.
        self.stop_control_steps = 0

    @property
    def use_tv(self) -> bool:
        return self.cfg.w_tv > 0

    @property
    def use_joints(self) -> bool:
        return self.cfg.w_joint > 0 or self.cfg.w_hinge > 0

    @property
    def enabled(self) -> bool:
        return self.use_tv or self.use_joints

    def mark_dirty(self):
        """Call after any op that reorders / resizes fg Gaussians."""
        self.dirty = True
        self.bones_dirty = True

    # ---------------------------------------------------------------- graph
    def _maybe_rebuild_knn(self, model, epoch: int):
        n = model.num_fg_gaussians
        reason = None
        if self.edges is None:
            reason = "init"
        elif self.dirty:
            reason = "densify/cull"
        elif self.edges.numel() > 0 and int(self.edges.max()) >= n:
            reason = "stale-index guard"
        elif epoch - self.last_knn_epoch >= self.cfg.knn_every_epochs:
            reason = "periodic"
        if reason is None:
            return
        means = model.fg.params["means"]
        self.edges, self.knn_med = build_knn_edges(
            means, self.cfg.tv_k, self.cfg.tv_max_dist_factor
        )
        self.fg_scale = robust_extent(means)
        self.last_knn_epoch = epoch
        self.dirty = False
        if reason != "periodic":
            # Indices changed (init / densify / cull / guard). A periodic
            # rebuild keeps Gaussian indices, so bones follow their own
            # bone_every_epochs schedule.
            self.bones_dirty = True
        guru.info(
            f"[prior] rebuilt kNN graph ({reason}): N={n} E={self.edges.shape[0]} "
            f"med_dist={self.knn_med:.4g} fg_scale={self.fg_scale:.4g}"
        )

    def _bg_kdtree(self, model) -> cKDTree | None:
        """Fresh tree each call: bg means keep training, so a tree cached by
        count would go stale. Only called from rebuild_bones (rare)."""
        if not model.has_bg:
            return None
        bg = model.bg.params["means"].detach().float().cpu().numpy()
        return cKDTree(bg)

    @torch.no_grad()
    def rebuild_bones(self, model) -> Bones:
        cfg = self.cfg
        w = model.fg.get_coefs()
        means = model.fg.params["means"]
        bones = extract_bones(
            w, self.edges, means, cfg.bone_tau, cfg.bone_n_min, cfg.adj_min_edges
        )
        if cfg.static_contact and model.has_bg and bones.num_bones > 0:
            T = self.num_frames
            ts = torch.linspace(0, T - 1, steps=min(cfg.contact_frames, T))
            ts = ts.round().long().unique().to(means.device)
            traj, _ = model.compute_poses_fg(ts)  # (N, S, 3)
            ids, cnts, cents = static_contacts(
                bones,
                traj,
                means,
                self._bg_kdtree(model),
                cfg.contact_dist_factor * self.knn_med,
                cfg.adj_min_edges,
            )
            if ids.numel() > 0:
                st = torch.stack([ids, torch.full_like(ids, STATIC)], dim=1)
                bones.pairs = torch.cat([bones.pairs, st], 0)
                bones.pair_counts = torch.cat([bones.pair_counts, cnts], 0)
                bones.centers = torch.cat([bones.centers, cents.float()], 0)
            elif not self._warned_no_contact:
                self._warned_no_contact = True
                guru.warning(
                    "[prior] static_contact is on but no bone touches the "
                    "background (contact radius "
                    f"{cfg.contact_dist_factor * self.knn_med:.4g}); no static "
                    "joints this scene. Static pivots inside the fg mask are "
                    "still handled as fg-fg joints."
                )
        return bones

    def _maybe_rebuild_bones(self, model, epoch: int, global_step: int | None):
        due = epoch - self.last_bone_epoch >= self.cfg.bone_every_epochs
        if not (self.bones is None or self.bones_dirty or due):
            return
        if global_step is not None and global_step < self.stop_control_steps:
            guru.warning(
                "[prior] extracting bones while densify/cull is still active; "
                "they will be re-extracted after every control step"
            )
        self.bones = self.rebuild_bones(model)
        self.last_bone_epoch = epoch
        self.bones_dirty = False
        b = self.bones
        n_static = int((b.pairs[:, 1] == STATIC).sum()) if b.pairs.numel() else 0
        guru.info(
            f"[prior] bones: M={b.num_bones} valid={int(b.valid.sum())} "
            f"pairs={b.pairs.shape[0]} (static={n_static}) "
            f"largest={b.sizes.sort(descending=True).values[:8].tolist()}"
        )

    # ----------------------------------------------------------------- loss
    def joint_terms(self, model) -> tuple[JointSolution, Tensor, Tensor]:
        """Returns (solution, normalised residual, joint class) for all pairs."""
        cfg = self.cfg
        b = self.bones
        T = self.num_frames
        w = model.fg.get_coefs()
        wbar = bone_mean_weights(w, b.labels, b.num_bones)
        ts = torch.arange(T, device=w.device)
        tf = model.motion_bases.compute_transforms(ts, wbar)  # (M, T, 3, 4)
        R, t = tf[..., :3], tf[..., 3]
        ia, ib = b.pairs[:, 0], b.pairs[:, 1]
        static = ib == STATIC
        Ra, ta = R[ia], t[ia]
        ib_safe = ib.clamp_min(0)
        Rb, tb = R[ib_safe], t[ib_safe]
        if static.any():
            # Static background: "point p of bone a stays fixed in world",
            # min_{p,q} mean_t ||R_a(t) p + t_a(t) - q||^2. Eliminating q
            # (q* = Rbar p + tbar) gives the same 3x3 solve with the partner
            # transform replaced by the temporal mean (Rbar, tbar). It does
            # not assume canonical == world, has no privileged frame, and
            # keeps the hinge null space: (R_t - Rbar) n = 0 iff all R_t share
            # the axis. Eigenvalues are ~half those of a moving-pair hinge.
            Rs = Ra.mean(dim=1, keepdim=True).expand_as(Ra)
            tsr = ta.mean(dim=1, keepdim=True).expand_as(ta)
            m = static[:, None, None, None]
            Rb = torch.where(m, Rs, Rb)
            tb = torch.where(static[:, None, None], tsr, tb)
        beta = cfg.beta_rel * T
        sol = solve_joint(Ra, ta, Rb, tb, b.centers, beta)
        res_n = sol.res / (self.fg_scale**2)
        cls = classify_joint(
            sol.evals, res_n.detach(), cfg.rot_eps, cfg.res_eps, cfg.hinge_ratio
        )
        return sol, res_n, cls

    def loss(
        self, model, epoch: int, global_step: int | None = None
    ) -> tuple[Tensor, dict]:
        cfg = self.cfg
        device = model.fg.params["means"].device
        total = torch.zeros((), device=device)
        stats: dict = {}
        want_tv = self.use_tv and epoch >= cfg.tv_start_epoch
        want_joint = self.use_joints and epoch >= cfg.joint_start_epoch
        if not (want_tv or want_joint):
            return total, stats

        self._maybe_rebuild_knn(model, epoch)
        stats["train/prior_num_edges"] = float(self.edges.shape[0])

        if want_tv:
            tv = charbonnier_tv(model.fg.get_coefs(), self.edges, cfg.tv_eps)
            total = total + cfg.w_tv * tv
            stats["train/tv_loss"] = tv.item()

        if want_joint:
            self._maybe_rebuild_bones(model, epoch, global_step)
            stats["train/num_bones"] = float(self.bones.num_bones)
            stats["train/num_valid_bones"] = float(self.bones.valid.sum())
            stats["train/num_joint_pairs"] = float(self.bones.pairs.shape[0])
            if self.bones.pairs.shape[0] > 0:
                sol, res_n, cls = self.joint_terms(model)
                jl = geman_mcclure(res_n, cfg.gm_c).mean()
                total = total + cfg.w_joint * jl
                stats["train/joint_loss"] = jl.item()
                is_h = cls == JOINT_HINGE
                if cfg.w_hinge > 0 and is_h.any():
                    hl = sol.hinge[is_h].mean()
                    total = total + cfg.w_hinge * hl
                    stats["train/hinge_loss"] = hl.item()
                for k, name in enumerate(JOINT_NAMES):
                    stats[f"train/joints_{name}"] = float((cls == k).sum())
        return total, stats
