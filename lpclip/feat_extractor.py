import os, argparse
import numpy as np
import torch
import sys

sys.path.append(os.path.abspath(".."))

from datasets.oxford_pets import OxfordPets
from datasets.oxford_flowers import OxfordFlowers
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.dtd import DescribableTextures
from datasets.eurosat import EuroSAT
from datasets.stanford_cars import StanfordCars
from datasets.food101 import Food101
from datasets.sun397 import SUN397
from datasets.caltech101 import Caltech101
from datasets.ucf101 import UCF101
from datasets.imagenet import ImageNet
from datasets.imagenetv2 import ImageNetV2
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenet_a import ImageNetA
from datasets.imagenet_r import ImageNetR

from dass.utils import setup_logger, set_random_seed, collect_env_info
from dass.config import get_cfg_default
from dass.data.transforms import build_transform
from dass.data import DatasetWrapper

import clip


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN

    # Existing OURS block
    cfg.TRAINER.OURS = CN()
    cfg.TRAINER.OURS.N_CTX = 10
    cfg.TRAINER.OURS.CSC = False
    cfg.TRAINER.OURS.CTX_INIT = ""
    cfg.TRAINER.OURS.WEIGHT_U = 0.1

    # Ensure dataset extras exist even if YAML doesn’t set them
    if not hasattr(cfg, "DATASET"):
        cfg.DATASET = CN()
    if not hasattr(cfg.DATASET, "SUBSAMPLE_CLASSES"):
        cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # compatible default ("all" | "base" | "new")


def setup_cfg(args):
    cfg = get_cfg_default()

    # allow unknown keys from YAMLs
    try:
        cfg.set_new_allowed(True)  # yacs>=0.1.8
    except AttributeError:
        from yacs.config import CfgNode as CN
        def _allow_new(n):
            if isinstance(n, CN):
                n.__dict__['_new_allowed'] = True
                for v in n.values():
                    _allow_new(v)
        _allow_new(cfg)

    extend_cfg(cfg)  # safe to keep

    # 1) dataset YAML, then 2) method YAML
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3) CLI overrides
    reset_cfg(cfg, args)

    # --- ENSURE REQUIRED DATASET KEY EXISTS (do this *after* merges/resets) ---
    cfg.defrost()
    try:
        _ = cfg.DATASET.SUBSAMPLE_CLASSES
    except AttributeError:
        cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # valid values typically: "all" | "base" | "new"
    cfg.freeze()
    # --------------------------------------------------------------------------

    return cfg



def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    # Device-safe setup
    device = "cuda" if (torch.cuda.is_available() and cfg.USE_CUDA) else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    ######################################
    #   Setup DataLoader
    ######################################
    dataset = eval(cfg.DATASET.NAME)(cfg)

    if args.split == "train":
        dataset_input = dataset.train_x
    elif args.split == "val":
        dataset_input = dataset.val
    else:
        dataset_input = dataset.test

    tfm_eval = build_transform(cfg, is_train=False)
    pin = (device == "cuda")
    data_loader = torch.utils.data.DataLoader(
        DatasetWrapper(cfg, dataset_input, transform=tfm_eval, is_train=False),
        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        sampler=None,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=pin,
    )

    ########################################
    #   Setup Network
    ########################################
    clip_model, _ = clip.load("RN50", device, jit=False)
    clip_model.eval()

    ###################################################################################################################
    # Start Feature Extractor
    feature_list = []
    label_list = []

    with torch.no_grad():
        for batch in data_loader:
            imgs = batch["img"].to(device, non_blocking=(device == "cuda"))
            feats = clip_model.visual(imgs).cpu().numpy()
            feature_list.extend(feats.tolist())
            label_list.extend(batch["label"].tolist())

    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME)
    os.makedirs(save_dir, exist_ok=True)
    save_filename = f"{args.split}"
    np.savez(
        os.path.join(save_dir, save_filename),
        feature_list=feature_list,
        label_list=label_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--num-shot", type=int, default=1, help="number of shots")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="which split")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    args = parser.parse_args()
    main(args)
