#!/usr/bin/env python3
"""Visualize CLIP feature spaces with t-SNE for different trained heads.

This script reuses the data pipeline defined in the repository to load a
subset of a dataset split, extracts features for a set of models (vanilla CLIP
and optional fine-tuned heads), projects them to 2D with t-SNE, and saves a
side-by-side comparison figure.

Example usage:

```
python scripts/plot_tsne.py \
  --root /path/to/datasets \
  --dataset-config-file configs/datasets/oxford_pets.yaml \
  --config-file configs/trainers/CoOp/rn50.yaml \
  --include-vanilla \
  --arcface-checkpoint /path/to/RN50_ArcFaceSigmoid.pth \
  --siglip-checkpoint /path/to/RN50_SigLipHead.pth \
  --samples-per-class 20 \
  --split test \
  --output tsne_comparison.png
```
"""

from __future__ import annotations

import argparse
import copy
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torchvision import transforms

from dass.config import get_cfg_default
from dass.engine import build_trainer
from loss.head.head_def import HeadFactory
from utils.util import Model


_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _build_clip_backbone(clip_visual: nn.Module) -> nn.Sequential:
    """Wrap a CLIP visual encoder with CLIP-style normalization."""

    normalize = transforms.Normalize(_CLIP_MEAN, _CLIP_STD)
    return nn.Sequential(normalize, clip_visual)


def _load_surrogate(
    clip_visual: nn.Module,
    num_classes: int,
    feature_dim: int,
    head_name: str,
    checkpoint_path: str,
    device: torch.device,
) -> Model:
    """Instantiate a surrogate head model and load weights from a checkpoint."""

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    backbone = _build_clip_backbone(copy.deepcopy(clip_visual))
    head_config = {"num_classes": num_classes, "output_dim": feature_dim}
    head_factory = HeadFactory(head_name, head_config)
    model = Model(backbone, head_factory)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading {checkpoint_path}: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading {checkpoint_path}: {unexpected}")

    model.to(device)
    model.eval()
    return model


def _parse_class_filter(
    raw_values: Optional[Sequence[str]], classnames: Sequence[str]
) -> Optional[List[int]]:
    if not raw_values:
        return None

    name_to_idx = {name.lower(): idx for idx, name in enumerate(classnames)}
    indices: List[int] = []
    for value in raw_values:
        value = value.strip()
        if not value:
            continue
        if value.isdigit():
            indices.append(int(value))
            continue
        key = value.lower()
        if key not in name_to_idx:
            raise ValueError(
                f"Unknown class '{value}'. Valid options include: {list(name_to_idx.keys())}"
            )
        indices.append(name_to_idx[key])
    if not indices:
        return None
    return indices


def _collect_samples(
    dataloader,
    num_classes: int,
    max_samples: Optional[int] = None,
    samples_per_class: Optional[int] = None,
    class_filter: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize a subset of images (on CPU) and the corresponding labels."""

    selected_images: List[torch.Tensor] = []
    selected_labels: List[torch.Tensor] = []
    counts = defaultdict(int)

    if class_filter is None:
        target_classes = list(range(num_classes))
    else:
        target_classes = list(class_filter)

    for batch in dataloader:
        images = batch["img"]
        labels = batch["label"]

        for img, label in zip(images, labels):
            label_int = int(label)
            if class_filter is not None and label_int not in class_filter:
                continue

            if samples_per_class is not None and counts[label_int] >= samples_per_class:
                continue

            selected_images.append(img.unsqueeze(0))
            selected_labels.append(torch.tensor([label_int], dtype=torch.long))
            counts[label_int] += 1

            if max_samples is not None and len(selected_labels) >= max_samples:
                break

        if max_samples is not None and len(selected_labels) >= max_samples:
            break

        if samples_per_class is not None:
            if all(counts.get(cls, 0) >= samples_per_class for cls in target_classes):
                break

    if not selected_images:
        raise RuntimeError("No samples were collected with the provided filters.")

    images_tensor = torch.cat(selected_images, dim=0)
    labels_tensor = torch.cat(selected_labels, dim=0)
    return images_tensor, labels_tensor


def _extract_features(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
    feature_stage: str,
) -> torch.Tensor:
    """Extract features (either backbone activations or logits)."""

    outputs: List[torch.Tensor] = []
    total = images.shape[0]
    model.eval()

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_images = images[start:end].to(device)
            batch_labels = labels[start:end].to(device)

            if feature_stage == "backbone":
                if hasattr(model, "backbone"):
                    feats = model.backbone(batch_images)
                else:
                    feats = model(batch_images)
            elif feature_stage == "logits":
                if hasattr(model, "head_type") and model.head_type != "fc":
                    feats = model(batch_images, batch_labels)
                else:
                    feats = model(batch_images)
            else:
                raise ValueError(f"Unsupported feature stage: {feature_stage}")

            outputs.append(feats.detach().float().cpu())

    return torch.cat(outputs, dim=0)


def _run_tsne(
    features_by_model: Dict[str, torch.Tensor],
    perplexity: float,
    random_state: int,
) -> Dict[str, np.ndarray]:
    """Apply t-SNE to each set of features."""

    embeddings: Dict[str, np.ndarray] = {}
    for name, feats in features_by_model.items():
        print(f"Running t-SNE for {name} on {feats.shape[0]} samples...")
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        )
        embeddings[name] = tsne.fit_transform(feats.numpy())
    return embeddings


def _plot_tsne(
    tsne_embeddings: Dict[str, np.ndarray],
    labels: np.ndarray,
    classnames: Sequence[str],
    output_path: str,
    cmap: str,
    point_size: float,
    alpha: float,
    show_legend: bool,
    dpi: int,
):
    """Create a side-by-side scatter plot for the provided t-SNE embeddings."""

    if not tsne_embeddings:
        raise ValueError("No embeddings to plot.")

    model_names = list(tsne_embeddings.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    cmap_obj = plt.cm.get_cmap(cmap, len(classnames))
    unique_labels = np.unique(labels)
    legend_handles: Optional[Sequence] = None
    legend_labels: Optional[Sequence[str]] = None

    for ax, model_name in zip(axes, model_names):
        embedding = tsne_embeddings[model_name]
        for cls in unique_labels:
            mask = labels == cls
            if not np.any(mask):
                continue
            color = cmap_obj(int(cls) % cmap_obj.N)
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=point_size,
                alpha=alpha,
                label=classnames[int(cls)],
                color=color,
            )
        ax.set_title(model_name)
        ax.set_xticks([])
        ax.set_yticks([])
        if show_legend and legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        else:
            ax.legend().remove() if ax.get_legend() else None

    if show_legend and legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(5, len(legend_labels)),
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE comparison to {output_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="t-SNE visualization for CLIP heads")
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/oxford_pets.yaml",
        help="Dataset configuration file",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/trainers/CoOp/rn50.yaml",
        help="Trainer configuration file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "sub-test", "sub-mf"],
        help="Dataset split to visualize",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device (e.g., 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for feature extraction",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to visualize",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Number of samples to draw per class",
    )
    parser.add_argument(
        "--class-filter",
        type=str,
        nargs="*",
        default=None,
        help="Optional class names or indices to visualize",
    )
    parser.add_argument(
        "--include-vanilla",
        action="store_true",
        help="Include vanilla CLIP (zeroshot) features",
    )
    parser.add_argument(
        "--arcface-checkpoint",
        type=str,
        default=None,
        help="Checkpoint for ArcFaceSigmoid head",
    )
    parser.add_argument(
        "--siglip-checkpoint",
        type=str,
        default=None,
        help="Checkpoint for SigLipHead",
    )
    parser.add_argument(
        "--feature-stage",
        type=str,
        default="backbone",
        choices=["backbone", "logits"],
        help="Which representation to feed into t-SNE",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for t-SNE",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tsne_comparison.png",
        help="Output image path",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="tab20",
        help="Matplotlib colormap name",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=10.0,
        help="Scatter point size",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Scatter point alpha",
    )
    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="Display class legend above the figure",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI")
    parser.add_argument(
        "--trainer",
        type=str,
        default="ZeroshotCLIP",
        help="Trainer name (passed to config)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="RN50",
        help="Backbone name (passed to config)",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Additional config options",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = str(device)
    if args.opts is None:
        args.opts = []

    cfg = get_cfg_default()
    cfg.merge_from_file(args.dataset_config_file)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.DATASET.ROOT = args.root
    cfg.MODEL.BACKBONE.NAME = args.backbone
    cfg.TRAINER.NAME = args.trainer
    cfg.device = str(device)
    cfg.freeze()

    # Setup trainer (data + CLIP model)
    trainer = build_trainer(cfg)
    dataloader_map = {
        "train": trainer.train_loader_x,
        "val": getattr(trainer, "val_loader", None),
        "test": trainer.test_loader,
        "sub-test": getattr(trainer, "sub_test_loader", None),
        "sub-mf": getattr(trainer, "sub_mf_loader", None),
    }
    if args.split not in dataloader_map or dataloader_map[args.split] is None:
        raise ValueError(f"Split '{args.split}' is not available in the data manager.")

    dataloader = dataloader_map[args.split]
    classnames = trainer.dm.dataset.classnames
    class_filter = _parse_class_filter(args.class_filter, classnames)
    images, labels = _collect_samples(
        dataloader,
        num_classes=trainer.dm.num_classes,
        max_samples=args.max_samples,
        samples_per_class=args.samples_per_class,
        class_filter=class_filter,
    )
    print(f"Collected {images.shape[0]} samples for visualization.")

    clip_visual = trainer.clip_model.visual.cpu()
    feature_dim = getattr(clip_visual, "output_dim", images.shape[1])

    scenarios: List[Tuple[str, nn.Module]] = []
    if args.include_vanilla:
        vanilla_model = _build_clip_backbone(copy.deepcopy(clip_visual)).to(device)
        vanilla_model.eval()
        scenarios.append(("Vanilla CLIP", vanilla_model))

    if args.arcface_checkpoint:
        arcface_model = _load_surrogate(
            clip_visual,
            trainer.dm.num_classes,
            feature_dim,
            "ArcFaceSigmoid",
            args.arcface_checkpoint,
            device,
        )
        scenarios.append(("ArcFaceSigmoid", arcface_model))

    if args.siglip_checkpoint:
        siglip_model = _load_surrogate(
            clip_visual,
            trainer.dm.num_classes,
            feature_dim,
            "SigLipHead",
            args.siglip_checkpoint,
            device,
        )
        scenarios.append(("SigLipHead", siglip_model))

    if not scenarios:
        raise ValueError("No models specified. Enable --include-vanilla or provide checkpoints.")

    features_by_model: Dict[str, torch.Tensor] = {}
    for name, model in scenarios:
        feats = _extract_features(
            model,
            images,
            labels,
            batch_size=args.batch_size,
            device=device,
            feature_stage=args.feature_stage,
        )
        features_by_model[name] = feats
        print(f"Extracted features for {name}: shape={tuple(feats.shape)}")

    tsne_embeddings = _run_tsne(
        features_by_model, perplexity=args.perplexity, random_state=args.random_state
    )
    _plot_tsne(
        tsne_embeddings,
        labels.numpy(),
        classnames,
        args.output,
        cmap=args.cmap,
        point_size=args.point_size,
        alpha=args.alpha,
        show_legend=args.show_legend,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    parser = build_argparser()
    main(parser.parse_args())
