"""Render the training-view videos (rgbs / depths / masks / normals / tracks)
for an already-trained run, i.e. what run_training.py saves into
<work_dir>/videos/epoch_XXXX/ -- without retraining.

Usage:
    python render_train_videos.py --work-dir outputs/running
"""
from dataclasses import dataclass

import torch
import tyro
import yaml
from loguru import logger as guru
from torch.utils.data import DataLoader

from flow3d.data import get_train_val_datasets
from flow3d.data.casual_dataset import CustomDataConfig, DavisDataConfig
from flow3d.data.iphone_dataset import iPhoneDataConfig
from flow3d.data.nvidia_dataset import NvidiaDataConfig
from flow3d.scene_model import SceneModel
from flow3d.validator import Validator


@dataclass
class Config:
    work_dir: str
    # which data loader the run used; inferred from cfg.yaml when None
    data_type: str | None = None


def guess_data_cls(data: dict, data_type: str | None):
    table = {
        "custom": CustomDataConfig,
        "davis": DavisDataConfig,
        "iphone": iPhoneDataConfig,
        "nvidia": NvidiaDataConfig,
    }
    if data_type is not None:
        return table[data_type]
    if "camera_type" in data and "mask_type" in data:
        return CustomDataConfig
    if data.get("camera_type") == "original":
        return NvidiaDataConfig
    raise ValueError("Could not infer data type; pass --data-type")


def main(cfg: Config):
    device = torch.device("cuda")
    with open(f"{cfg.work_dir}/cfg.yaml") as f:
        train_cfg = yaml.safe_load(f)

    data = train_cfg["data"]
    data_cls = guess_data_cls(data, cfg.data_type)
    fields = data_cls.__dataclass_fields__
    data = {k: v for k, v in data.items() if k in fields}
    data["load_from_cache"] = True  # reuse the scene normalization from training
    data_cfg = data_cls(**data)

    train_dataset, train_video_view, _, _ = get_train_val_datasets(
        data_cfg, load_val=False
    )
    assert train_video_view is not None, "this data type has no video view"

    ckpt = torch.load(f"{cfg.work_dir}/checkpoints/last.ckpt", weights_only=False)
    model = SceneModel.init_from_state_dict(ckpt["model"])
    model.use_2dgs = train_cfg.get("use_2dgs", False)
    model = model.to(device)
    epoch = ckpt.get("epoch", train_cfg.get("num_epochs", 500) - 1)
    guru.info(f"Loaded checkpoint at epoch {epoch}, step {ckpt.get('global_step')}")

    validator = Validator(
        model=model,
        device=device,
        train_loader=DataLoader(train_video_view, batch_size=1),
        val_img_loader=None,
        val_kpt_loader=None,
        save_dir=cfg.work_dir,
    )
    validator.save_train_videos(epoch)
    guru.info(f"Videos written to {cfg.work_dir}/videos/epoch_{epoch:04d}/")


if __name__ == "__main__":
    main(tyro.cli(Config))
