# Shape of Motion — Quick Reference

A step-by-step cheatsheet for preprocessing, training, rendering, and viewing.

## ⚙️ Environments

| Environment | Use for |
| --- | --- |
| `som` | All preprocessing (DROID-SLAM needs torch 2.2) |
| `som_train` | Training, rendering, and viewing |

## 📁 Step 1 — Place the video

```bash
mkdir -p ~/Desktop/3D_project/running/videos
mv ~/Desktop/3D_project/running/running.mp4 ~/Desktop/3D_project/running/videos/
```

Scripts expect the video at `<root>/videos/<name>.mp4`.

## 🖼️ Step 2 — Extract frames & draw mask

```bash
conda activate som
cd ~/Desktop/3D_project/shape-of-motion/preproc
python mask_app.py --root_dir /home/babu/Desktop/3D_project/running
```

Open the printed URL in your browser, then:

1. Pick `running.mp4` and extract frames.
2. Click the person on the first frame to segment them (Segment Anything).
3. Run tracking so the mask follows through every frame.
4. Save the masks.

✅ Creates: `running/images/running/` and `running/masks/running/`

## 🔧 Step 3 — Run all preprocessing

```bash
python process_custom.py --img-dirs /home/babu/Desktop/3D_project/running/images/running --gpus 0
```

| Step | Output folder |
| --- | --- |
| Metric depth (UniDepth) | `unidepth_disp/`, `unidepth_intrins/` |
| Relative depth (Depth Anything) | `depth_anything/`, `aligned_depth_anything/` |
| Camera poses (DROID-SLAM) | `droid_recon/running.npy` |
| 2D point tracks (BootsTAPIR) | `bootstapir/running/` |

⏳ Point tracking is the slowest step — it tracks between every pair of frames.

To re-run a single step individually (all in `preproc/`):

- `launch_metric_depth.py`
- `launch_depth.py`
- `launch_slam.py`
- `launch_tracks.py`

## 📋 Step 4 — Copy camera file

```bash
cp ~/Desktop/3D_project/running/droid_recon/running.npy ~/Desktop/3D_project/running/droid_recon.npy
```

The training loader reads `<root>/droid_recon.npy`, not the subfolder where step 3 saves it.

## 🚀 Step 5 — Train

```bash
conda activate som_train
cd ~/Desktop/3D_project/shape-of-motion
python run_training.py \
    --work-dir ./outputs/running \
    data:custom \
    --data.data-dir /home/babu/Desktop/3D_project/running \
    --data.res running \
    --data.camera-type droid_recon
```

⚠️ General flags go **before** `data:custom`. The `--data.*` flags go **after** it.

If `--work-dir` already has a checkpoint, training **continues from it** — use a new folder name for a fresh run. Run `python run_training.py data:custom --help` to list all options.

### Key training flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--num-epochs` | 500 | Main training length setting |
| `--batch-size` | 8 | Frames per training step |
| `--optim.max-steps` | 5000 | LR schedule length. Set ≈ epochs × (frames ÷ batch-size) |
| `--num-fg / --num-bg` | 40000 / 100000 | Starting Gaussians for foreground / background |
| `--num-motion-bases` | 10 | Basic motion patterns used to describe movement |
| `--validate-every / --save-videos-every` | 50 | Epochs between evaluations / video saves |

**Steps formula:** total steps = epochs × (frames ÷ batch-size). E.g. 500 epochs × (77 frames ÷ 8) ≈ 5000 steps.

## 🎞️ Render training videos

```bash
conda activate som_train
python render_train_videos.py --work-dir outputs/running
```

Outputs to `outputs/running/videos/epoch_0499/` — six files: `rgbs`, `depths`, `masks`, `normals`, `tracks_2d`, `motion_coefs`. In `rgbs.mp4` the real frame is on the left and the model's render is on the right.

## 👁️ Interactive 3D viewer

```bash
conda activate som_train
cd ~/Desktop/3D_project/shape-of-motion
python run_rendering.py --work-dir outputs/running --port 8890
```

Open `http://localhost:8890` when it prints `Starting rendering`. Drag to rotate; use the time slider to play.

- Change `--work-dir` to switch sequences (`outputs/walking`, `outputs/skating_v1`, etc.)
- If the port is in use, try `--port 8891`
- On a remote machine via VS Code: forward the port in the **Ports** tab first

## 🗂️ Config & defaults

There is no config file that training reads. Settings are Python classes with defaults in:

- `run_training.py` lines 56–77 — epochs, batch size, number of Gaussians, etc.
- `flow3d/configs.py` — learning rates, loss weights, densification and optimizer settings.

The `cfg.yaml` saved in each output folder is a **read-only record** of what a run used — editing it has no effect.

## 🐛 Known issues

- **DROID-SLAM `torch_scatter` error:** already fixed by the added file — no action needed.
- Any other failure: paste the error. You can re-run just that step with the individual `launch_*.py` scripts in `preproc/`.














Use the `som` environment for every preprocessing step (DROID-SLAM there needs its torch 2.2). Then use `som_train` for training. I haven't run these steps for `running` myself, but they are the same ones that produced `mydata/walking`.

### 1. Put the video where the scripts expect it
The scripts need `<root>/videos/<name>.mp4`:
```bash
mkdir -p ~/Desktop/3D_project/running/videos
mv ~/Desktop/3D_project/running/running.mp4 ~/Desktop/3D_project/running/videos/
```

### 2. Extract frames and draw the mask (interactive)
```bash
conda activate som
cd ~/Desktop/3D_project/shape-of-motion/preproc
python mask_app.py --root_dir /home/babu/Desktop/3D_project/running
```
Open the URL it prints in your browser, then:
1. Pick `running.mp4` and extract the frames.
2. On the first frame, click on the person to select them, using Segment Anything.
3. Run tracking so the mask follows the person through every frame.
4. Save the masks.

This creates `running/images/running/` and `running/masks/running/`.

### 3. Run the rest of the preprocessing automatically
```bash
python process_custom.py --img-dirs /home/babu/Desktop/3D_project/running/images/running --gpus 0
```
This runs four steps in order:

| Step | Output folder |
|---|---|
| Metric depth (UniDepth) | `unidepth_disp/`, `unidepth_intrins/` |
| Relative depth (Depth Anything), aligned to the metric depth | `depth_anything/`, `aligned_depth_anything/` |
| Camera poses (DROID-SLAM) | `droid_recon/running.npy` |
| 2D point tracks (BootsTAPIR) | `bootstapir/running/` |

Point tracking is the slowest part, because it tracks between every pair of frames.

### 4. Copy the camera file to where training looks for it
In this repo, the training loader reads `<root>/droid_recon.npy`. That's a different place from where step 3 saves it:
```bash
cp ~/Desktop/3D_project/running/droid_recon/running.npy ~/Desktop/3D_project/running/droid_recon.npy
```

### 5. Train
```bash
conda activate som_train
cd ~/Desktop/3D_project/shape-of-motion
python run_training.py \
    --work-dir ./outputs/running \
    data:custom \
    --data.data-dir /home/babu/Desktop/3D_project/running \
    --data.res running \
    --data.camera-type droid_recon
```

**If a step fails:**
- The DROID-SLAM error about `torch_scatter` is already fixed by the file I added earlier.
- If any step fails, paste me the error. You can also run just that step on its own with `launch_depth.py`, `launch_metric_depth.py`, `launch_slam.py` or `launch_tracks.py`, all in `preproc/`.




There's no config file that training reads. The settings are written as Python classes with default values, and you can change any of them from the command line. The `cfg.yaml` saved in each output folder (for example [outputs/walking/cfg.yaml](outputs/walking/cfg.yaml)) is only a record of what a run used; editing it changes nothing.

The defaults live in two files:
- [run_training.py:56-77](run_training.py#L56-L77): number of epochs, batch size, number of Gaussians, and similar.
- [flow3d/configs.py](flow3d/configs.py): learning rates, loss weights, and densification and optimizer settings.

**Most useful settings**

| Flag | Default | What it does |
|---|---|---|
| `--num-epochs` | 500 | Length of training. This is the main training-length setting. |
| `--batch-size` | 8 | Frames per training step |
| `--optim.max-steps` | 5000 | Length of the learning-rate schedule. Set it to roughly the total steps (see below). |
| `--num-fg` / `--num-bg` | 40000 / 100000 | Starting number of Gaussians for the moving foreground / static background |
| `--num-motion-bases` | 10 | Number of basic motion patterns used to describe the movement |
| `--validate-every` / `--save-videos-every` | 50 | How often, in epochs, to evaluate and save videos |
| `--loss.w-rgb`, `--loss.w-track`, `--loss.w-depth-reg` … | see configs.py | Weights of the different loss terms |
| `--lr.fg.means`, `--lr.bg.colors` … | see configs.py | Learning rates |

**Training steps vs epochs:** total steps = epochs × (number of frames ÷ batch size). Walking had 77 frames ÷ 8 ≈ 10 steps per epoch, so 500 epochs came to 5000 steps. That matches the saved checkpoint's step count of 5000.

**Example: train twice as long**
```bash
python run_training.py \
    --work-dir ./outputs/walking_1000ep \
    --num-epochs 1000 \
    --optim.max-steps 10000 \
    data:custom \
    --data.data-dir /home/babu/Desktop/3D_project/shape-of-motion/mydata \
    --data.res walking \
    --data.camera-type droid_recon
```
Put the general flags **before** `data:custom`, and the `--data.*` flags after it.

**Things to know:**
- To list every option, run `python run_training.py data:custom --help`.
- If the `--work-dir` folder already has a checkpoint, training **continues from it** instead of starting over. So use a new folder name for a fresh run. On the other hand, raising `--num-epochs` on the same folder simply trains it further.
- The first stage (~1000 iterations, the "initialisation" progress bar you saw before) has no flag. It's set in [flow3d/init_utils.py:285](flow3d/init_utils.py#L285), and you'd only change that if you have a specific reason.



Running and walking now have their videos too. They weren't missing because something failed: the training script simply never saved videos for your own videos (the `custom` data type), only for dataset types like Skating's.

**What I changed:**
- **Rendered from your existing checkpoints, no retraining:**
  - [outputs/running/videos/epoch_0499/](outputs/running/videos/epoch_0499/)
  - [outputs/walking/videos/epoch_0499/](outputs/walking/videos/epoch_0499/)

  Each folder has the same six files as Skating: `rgbs`, `depths`, `masks`, `normals`, `tracks_2d` and `motion_coefs`, 77 frames at 30 fps. In `rgbs.mp4` the real frame is on the left and the model's render is on the right.
- **New script, [render_train_videos.py](render_train_videos.py):** builds these videos for any finished run:
  ```bash
  conda activate som_train
  python render_train_videos.py --work-dir outputs/<run_name>
  ```
- **Training fix:** [flow3d/data/__init__.py](flow3d/data/__init__.py) and [flow3d/data/casual_dataset.py](flow3d/data/casual_dataset.py). Future `data:custom` runs will save `videos/epoch_0050`, `epoch_0100`, and so on, like Skating. This adds roughly 10–20 seconds every 50 epochs.

These videos are rendered from the camera's own path, so they show how well the model reproduces the original video. To judge other viewing angles, use the viewer (`run_rendering.py`).





This command opens the interactive 3D viewer:

```bash
conda activate som_train
cd ~/Desktop/3D_project/shape-of-motion
python run_rendering.py --work-dir outputs/running --port 8890
```

When it prints `Starting rendering`, open **http://localhost:8890** in your browser. You can drag to rotate the view and use the time slider to play the motion.

- To view a different run, change `--work-dir` to `outputs/walking` or `outputs/skating_v1`.
- If port 8890 is already in use, pick another, e.g. `--port 8891`.
- If you're connected to this machine remotely through VS Code, forward the port in the **Ports** tab first.

To regenerate the videos instead, use `python render_train_videos.py --work-dir outputs/running`.