# Motion Bases in Shape of Motion: a code walkthrough

This guide explains how the **motion basis** idea is implemented in this repo, where each piece lives, and where to make changes for your own experiments.

All links are relative to this file (`docs/`). Line numbers match the code as of 2026-11-18.

---

## 1. The idea in one paragraph

The scene is made of 3D Gaussians.
- **Background** Gaussians are static.
- **Foreground** (moving) Gaussians are each stored **once**, in a *canonical* frame `t_c`.

Instead of learning a separate trajectory for every Gaussian (tens of thousands of trajectories), the model learns a small set of **K shared rigid motions over time**, the *motion bases*. Each Gaussian then learns **K mixing weights** saying how much it follows each basis.

A Gaussian on the left hand mostly follows the "left arm" basis. A Gaussian on the elbow is a blend of the "upper arm" and "forearm" bases. Because K is small (default 10), the motion is low-dimensional and therefore regularised: nearby, similar-moving points are forced to share motion.

---

## 2. The math

Notation:

| Symbol | Meaning | Shape in code |
|---|---|---|
| `G` | number of foreground Gaussians | |
| `K` | number of motion bases (`--num-motion-bases`, default 10) | |
| `T` | number of frames | |
| $R_k(t)$, $\mathbf{t}_k(t)$ | rotation / translation of basis k at frame t | `rots (K,T,6)`, `transls (K,T,3)` |
| $\mathbf{w}_i$ | mixing weights of Gaussian i | `motion_coefs (G,K)`, after softmax |
| $\boldsymbol{\mu}_i^{c}$, $q_i^{c}$ | canonical position / orientation of Gaussian i | `fg.params["means"]`, `fg.params["quats"]` |

**Step 1: per-Gaussian weights** (a softmax, so they are positive and sum to 1):

$$\mathbf{w}_i = \mathrm{softmax}(\mathbf{c}_i), \quad \mathbf{c}_i \in \mathbb{R}^K \text{ (learned logits)}$$

**Step 2: blend the bases.** This is a *linear* blend of the 6D rotation vectors and the translations:

$$\mathbf{r}_i(t) = \sum_k w_{ik}\,\mathbf{r}_k(t) \in \mathbb{R}^6, \qquad \mathbf{t}_i(t) = \sum_k w_{ik}\,\mathbf{t}_k(t)$$

**Step 3: turn the blended 6D vector back into a valid rotation** with Gram–Schmidt (`cont_6d_to_rmat`):

$$R_i(t) = \mathrm{GS}(\mathbf{r}_i(t)), \qquad T_i(t) = [R_i(t)\,|\,\mathbf{t}_i(t)] \in SE(3)$$

**Step 4: move the canonical Gaussian:**

$$\boldsymbol{\mu}_i(t) = R_i(t)\,\boldsymbol{\mu}_i^{c} + \mathbf{t}_i(t), \qquad q_i(t) = \mathrm{quat}(R_i(t)) \otimes q_i^{c}$$

Scales, colours and opacities do **not** change over time. Only position and orientation are animated.

> The blend is done in 6D-rotation space and then re-orthonormalised. This is *not* proper SE(3) interpolation (such as dual quaternions or log-space blending), but it is smooth, cheap and differentiable. Swapping it out is a natural experiment (see §7).

---

## 3. Where the code lives (map)

```
run_training.py
 └─ initialize_and_checkpoint_model()
     ├─ init_model_from_tracks()               run_training.py:219   ← chooses cano_t, rot_type
     │   ├─ init_motion_params_with_procrustes()  init_utils.py:115  ← builds initial bases + coefs
     │   │   └─ sample_initial_bases_centers()    init_utils.py:540  ← k-means on velocity directions
     │   └─ init_fg_from_tracks_3d()              init_utils.py:40   ← canonical Gaussians
     └─ run_initial_optim()                       init_utils.py:279  ← 1000-iter fit of bases to 3D tracks
 └─ training loop → Trainer.train_step()          trainer.py:167
     ├─ model.compute_poses_all(ts)               scene_model.py:110
     │   └─ compute_poses_fg()                    scene_model.py:78  ← Step 4
     │       └─ compute_transforms()              scene_model.py:69
     │           └─ MotionBases.compute_transforms()  params.py:232  ← Steps 2–3
     └─ losses on bases                           trainer.py:439-488
```

| What | File : line |
|---|---|
| `MotionBases` class (stores `rots`, `transls`) | [flow3d/params.py:212](../flow3d/params.py#L212) |
| Blending formula (Steps 2–3) | [flow3d/params.py:232](../flow3d/params.py#L232) |
| Softmax on coefficients (Step 1) | [flow3d/params.py:136](../flow3d/params.py#L136), [`get_coefs` :173](../flow3d/params.py#L173) |
| 6D → rotation matrix | [flow3d/transforms.py:41](../flow3d/transforms.py#L41) |
| Apply transform to Gaussians (Step 4) | [flow3d/scene_model.py:78](../flow3d/scene_model.py#L78) |
| Foreground and background combined for rendering | [flow3d/scene_model.py:110](../flow3d/scene_model.py#L110) |
| Initialisation (clustering + Procrustes) | [flow3d/init_utils.py:115](../flow3d/init_utils.py#L115) |
| Clustering of tracks | [flow3d/init_utils.py:540](../flow3d/init_utils.py#L540) |
| Initial fitting stage (1000 iterations) | [flow3d/init_utils.py:279](../flow3d/init_utils.py#L279) |
| Smoothness loss on bases | [flow3d/loss_utils.py:175](../flow3d/loss_utils.py#L175) |
| Learning rates of bases / coefs | [flow3d/configs.py:5-27](../flow3d/configs.py#L5-L27) |
| Optimizer construction | [flow3d/trainer.py:751](../flow3d/trainer.py#L751) |
| `motion_coefs.mp4` visualisation | [flow3d/validator.py:422](../flow3d/validator.py#L422) |

---

## 4. The core code, annotated

### 4.1 `MotionBases.compute_transforms`: [params.py:232](../flow3d/params.py#L232)

```python
def compute_transforms(self, ts, coefs):
    # ts: (B,) frame indices      coefs: (G, K) softmaxed weights
    transls = self.params["transls"][:, ts]                 # (K, B, 3)  basis translations at those frames
    rots    = self.params["rots"][:, ts]                    # (K, B, 6)  basis rotations (6D) at those frames
    transls = torch.einsum("pk,kni->pni", coefs, transls)   # (G, B, 3)  Σ_k w_ik t_k(t)
    rots    = torch.einsum("pk,kni->pni", coefs, rots)      # (G, B, 6)  Σ_k w_ik r_k(t)
    rotmats = cont_6d_to_rmat(rots)                         # (G, B, 3, 3) Gram-Schmidt
    return torch.cat([rotmats, transls[..., None]], dim=-1) # (G, B, 3, 4)
```

Important properties:
- The bases are a **lookup table per frame**: `rots[k, t]` is a free parameter for every frame. There is no continuous function of time, so you **cannot query between frames** (e.g. t = 3.5) without changing this.
- The transform maps **canonical → world at time t**. The canonical frame `t_c` gets the identity at initialisation, but it is not fixed to identity during training.

### 4.2 `SceneModel.compute_poses_fg`: [scene_model.py:78](../flow3d/scene_model.py#L78)

```python
means = self.fg.params["means"]                    # (G, 3) canonical positions
transfms = self.compute_transforms(ts, inds)       # (G, B, 3, 4)
means = einsum("pnij,pj->pni", transfms, pad(means, 1.0))   # R μ + t
quats = rotmat_to_quat(transfms[..., :3, :3]) ⊗ quats       # rotate orientation
```

### 4.3 Coefficients: [params.py:136](../flow3d/params.py#L136)

```python
self.motion_coef_activation = lambda x: F.softmax(x, dim=-1)
```

`motion_coefs` are stored as **logits** alongside each Gaussian's other parameters. When a Gaussian is split or duplicated during densification ([params.py:177](../flow3d/params.py#L177)), its `motion_coefs` are **copied**, so child Gaussians inherit the parent's motion.

---

## 5. How the bases are initialised

All of this happens before the main training loop. It's the `0.0xx 0.0xx … /1000` progress bar you see at the start of a run.

1. **Lift 2D tracks to 3D.** The BootsTAPIR tracks are lifted to 3D with depth and the cameras, giving `tracks_3d.xyz` of shape `(G, T, 3)` (`num_fg` = 40k tracks).
2. **Pick the canonical frame.** It's the frame where the most tracks are visible ([run_training.py:239](../run_training.py#L239)):
   `cano_t = visibles.sum(dim=0).argmax()`
3. **Cluster tracks by motion, not position** ([init_utils.py:540](../flow3d/init_utils.py#L540)):
   - fill in missing track points by linear interpolation
   - compute frame-to-frame **velocity directions** (unit vectors), flattened to `(G, (T-1)*3)`
   - run **k-means** (cuML, on GPU) with `K = num_motion_bases` clusters
   - the cluster centre = median canonical position of its members
4. **Drop small clusters.** Clusters with ≤ 100 tracks are removed ([init_utils.py:163](../flow3d/init_utils.py#L163)), so the **actual K can be smaller than `--num-motion-bases`**. Check the log line `num_bases=…`.
5. **Initial coefficients from distance** ([init_utils.py:171](../flow3d/init_utils.py#L171)):
   `motion_coefs = 10 * exp(-‖μ_i − center_k‖)` (logits, so after softmax nearer clusters dominate).
6. **Procrustes per cluster per frame.** For each cluster and each frame t, solve the best rigid transform mapping the cluster's points at `t_c` to their points at `t`, weighted by visibility × confidence. Frames with too little support copy the previous frame's transform.
7. **`run_initial_optim`** ([init_utils.py:279](../flow3d/init_utils.py#L279)): 1000 Adam steps that fit `rots`, `transls`, `motion_coefs` and the canonical `means` to the 3D tracks. Losses used:

   | Loss | Purpose | Weight |
   |---|---|---|
   | 3D track L1 | positions match lifted tracks | 1.0 |
   | 2D reprojection L1 | match tracks in image space | 0.5 |
   | coef sparsity `1 − Σ w²` | push each Gaussian toward one basis | 0.01 |
   | basis smoothness (accel) | smooth basis trajectories | 0.01 → 0.1 |
   | track smoothness (accel) | smooth Gaussian trajectories | half of the above |
   | z-acceleration | little acceleration along the camera ray | 0.1 |

---

## 6. How the bases are trained (main loop)

During `Trainer.train_step` ([trainer.py:167](../flow3d/trainer.py#L167)), the bases get gradients from **every** loss through `compute_poses_all`: RGB, depth, mask and 2D tracks. Basis-specific terms:

| Term | Code | CLI flag (default) |
|---|---|---|
| Basis smoothness: central-difference acceleration of `rots` (×1) and `transls` (×2) | [trainer.py:439](../flow3d/trainer.py#L439), [loss_utils.py:175](../flow3d/loss_utils.py#L175) | `--loss.w-smooth-bases` (0.1) |
| Per-Gaussian trajectory smoothness | [trainer.py:457](../flow3d/trainer.py#L457) | `--loss.w-smooth-tracks` (2.0) |
| Acceleration along the viewing ray | [trainer.py:485](../flow3d/trainer.py#L485) | `--loss.w-z-accel` (1.0) |
| 2D track loss (main motion supervision) | [trainer.py:401](../flow3d/trainer.py#L401) | `--loss.w-track` (2.0) |

Learning rates ([configs.py](../flow3d/configs.py)), all Adam with a constant rate:

| Param | Flag | Default |
|---|---|---|
| basis rotations | `--lr.motion-bases.rots` | 1.6e-4 |
| basis translations | `--lr.motion-bases.transls` | 1.6e-4 |
| per-Gaussian coefficients | `--lr.fg.motion-coefs` | 1e-2 |

Note: the coefficient sparsity loss exists **only** in the initial fitting stage, not in the main loop.

---

## 7. Experiments: what to change and where

### A. Without touching code (CLI only)

| Experiment | Command fragment | What to look for |
|---|---|---|
| Fewer / more bases | `--num-motion-bases 4` / `20` | Too few makes limbs blur together; too many gives jittery or overfit motion. Check the actual K in the log. |
| Freeze bases after init | `--lr.motion-bases.rots 0 --lr.motion-bases.transls 0` | How much the main loop refines the motion |
| Freeze coefficients | `--lr.fg.motion-coefs 0` | Whether the weights need to adapt |
| Smoothness ablation | `--loss.w-smooth-bases 0` / `1.0` | Jitter vs. over-smoothed motion |
| Turn off trajectory smoothness | `--loss.w-smooth-tracks 0` | |
| Weaker track supervision | `--loss.w-track 0.5` | How much the model relies on the 2D tracks |

Use a new `--work-dir` for each experiment. Training resumes if the folder already has a checkpoint.

### B. Small code changes

1. **Clustering method.** `init_motion_params_with_procrustes(..., cluster_init_method="kmeans")` also supports `"hdbscan"`, which picks K automatically ([init_utils.py:120](../flow3d/init_utils.py#L120), called at [run_training.py:241](../run_training.py#L241)). You could also cluster on position + velocity instead of velocity only (edit `vel_dirs` at [init_utils.py:578](../flow3d/init_utils.py#L578)).
2. **Coefficient activation** ([params.py:136](../flow3d/params.py#L136)):
   - softmax with a temperature: `softmax(x / τ)`
   - sparsemax or top-k, so each Gaussian uses at most k bases
   - plain linear (unnormalised) weights
3. **Add coefficient sparsity to the main loop.** Copy `1 - (coefs**2).sum(-1).mean()` from [init_utils.py:375](../flow3d/init_utils.py#L375) into `train_step` with a new `LossesConfig` weight.
4. **Spatial smoothness of coefficients.** Penalise nearest neighbours in canonical space having different `motion_coefs` (kNN graph loss). This encourages piecewise-rigid parts.
5. **Rotation parameterisation.** `rot_type = "6d"` at [run_training.py:238](../run_training.py#L238); `"quat"` is supported in the init code, but `compute_transforms` assumes 6D. You would also need to change `cont_6d_to_rmat` there.

### C. Architectural changes

| Idea | Where to change | Notes |
|---|---|---|
| **Proper SE(3) blending** (dual quaternions, or blending in log space `exp(Σ w log T)`) | `MotionBases.compute_transforms`, [params.py:232](../flow3d/params.py#L232) | Keep the same signature: `(ts, coefs) → (G, B, 3, 4)`. Nothing else needs to change. |
| **Continuous-time bases** (time MLP, B-splines, Fourier series) instead of a per-frame table | `MotionBases.__init__` + `compute_transforms` | Lets you render at fractional t and reduces parameters. You'll need to initialise from the Procrustes result, e.g. by fitting the spline to `init_rots` / `init_ts`. |
| **Time-varying coefficients** `w_i(t)` | `GaussianParams` (coefs become `(G, T, K)` or an MLP) + `SceneModel.compute_transforms` | More expressive; needs a strong smoothness prior |
| **Hierarchical bases** (global body motion ∘ local part motion) | `MotionBases.compute_transforms` | Compose two transforms: `T_global(t) · T_local,i(t)` |
| **Non-rigid residual** per Gaussian | `SceneModel.compute_poses_fg`, [scene_model.py:78](../flow3d/scene_model.py#L78) | Add a small learned offset `δ_i(t)` after the rigid transform; regularise it toward zero |
| **Animate scales too** | `compute_poses_fg` + renderer call sites | Currently only means and quats move |

**Rules to keep the rest of the pipeline working:**
- `MotionBases.params` must stay an `nn.ParameterDict`. The optimizer builder ([trainer.py:751](../flow3d/trainer.py#L751)) creates one Adam optimizer per parameter named `motion_bases.params.<field>`, and looks up its learning rate in `SceneLRConfig.motion_bases.<field>`. A **new parameter needs a matching field in `MotionLRConfig`** ([configs.py:24](../flow3d/configs.py#L24)).
- Anything per-Gaussian (stored in `fg.params`) is automatically duplicated, split and culled during densification ([params.py:177-200](../flow3d/params.py#L177-L200)), but only if its first dimension is `G`.
- Checkpoint loading uses `init_from_state_dict` ([params.py:226](../flow3d/params.py#L226), [scene_model.py:148](../flow3d/scene_model.py#L148)). Update it if you add or rename parameters, or old checkpoints won't load.
- `run_initial_optim` calls `bases.compute_transforms` directly, so a new blending method is used there too.

---

## 8. How to inspect the bases

**`motion_coefs.mp4`** (in `outputs/<run>/videos/epoch_XXXX/`): each foreground Gaussian is coloured by a 3-component PCA of its coefficient vector, and the background is grey. Same colour means same motion group, so you should see body parts as coherent colour regions.

**From a checkpoint** (run in `som_train`):

```python
import torch
from flow3d.scene_model import SceneModel

ckpt = torch.load("outputs/walking/checkpoints/last.ckpt", weights_only=False)
model = SceneModel.init_from_state_dict(ckpt["model"])

rots = model.motion_bases.params["rots"]      # (K, T, 6)
transls = model.motion_bases.params["transls"] # (K, T, 3)
coefs = model.fg.get_coefs()                  # (G, K), softmaxed

print("K =", rots.shape[0], " T =", rots.shape[1], " G =", coefs.shape[0])
print("Gaussians per dominant basis:", torch.bincount(coefs.argmax(-1), minlength=rots.shape[0]))
print("mean max-weight (1.0 = each Gaussian uses one basis):", coefs.max(-1).values.mean().item())

# trajectory of basis k's translation over time → plot with matplotlib
import matplotlib.pyplot as plt
for k in range(rots.shape[0]):
    plt.plot(transls[k, :, 0].detach().cpu(), label=f"basis {k}")
plt.legend(); plt.xlabel("frame"); plt.ylabel("x translation"); plt.savefig("bases_x.png")
```

Example output for the `walking` run (2026-11-18):

```
K = 10  T = 77  G = 47640
Gaussians per dominant basis: [5211, 2528, 1944, 1459, 2293, 8664, 17961, 876, 6496, 208]
mean max-weight: 0.508
```

How to read this:
- All 10 bases survived initialisation.
- Basis 6 dominates (~38% of Gaussians). It's probably the torso / overall body movement.
- Basis 9 is almost unused.
- A mean max-weight of ~0.5 means a typical Gaussian blends about two bases rather than following one rigidly.

These are good baselines to compare against when you change K, the activation function or the sparsity term.

Useful things to measure across experiments:
- **Coefficient entropy:** how many bases each Gaussian really uses.
- **Basis usage:** whether some bases are dead (almost no Gaussians use them).
- **Basis acceleration:** how jittery the motion is.
- **Training metrics:** `train/fg_psnr` and the 2D track loss in TensorBoard.
- **Validation metrics:** `val/psnr`, `val/lpips` for datasets with test cameras (e.g. Skating).
