import torch
from .enumType import NormType
from torchvision import transforms
import timm
# from models import *
from dass.config import get_cfg_default
import ruamel.yaml as yaml
from pathlib import Path
from torchvision.models import ResNet101_Weights, ResNet50_Weights, ViT_B_16_Weights, MobileNet_V2_Weights, EfficientNet_B0_Weights, DenseNet121_Weights
from torchvision._internally_replaced_utils import load_state_dict_from_url
import torch.nn.functional as F
import torch.nn as nn
import random

normalize_list = {'clip': transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                  'general': transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                  'imagenet': transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}

def clamp_by_l2(x, max_norm):
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor

def random_init(x, norm_type, epsilon):
    delta = torch.zeros_like(x)
    if norm_type == NormType.Linf:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == NormType.L2:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta


def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def read_yaml(path):
    return yaml.load(open(path, 'r'), Loader=yaml.Loader)

def dir_check(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def distance(A, B):
    prod = A @ B.T

    prod_A = A @ A.T
    norm_A = prod_A.diag().unsqueeze(1).expand_as(prod)

    prod_B = B @ B.T
    norm_B = prod_B.diag().unsqueeze(0).expand_as(prod)

    res = norm_A + norm_B - 2 * prod
    return res

def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d' % (orig_size ** 2, new_size ** 2))

        return new_pos_embed
    else:
        return pos_embed_checkpoint


class Model(nn.Module):
    def __init__(self, backbone_factory, head_factory):
        super(Model, self).__init__()
        self.backbone = backbone_factory
        self.head = head_factory.get_head()
        self.head_type = head_factory.head_type

    def forward(self, data, label=None, return_features = False):
        feat = self.backbone(data)
        if self.head_type == 'fc':
            pred = self.head(feat)
        else:
            pred = self.head(feat, label)

        if not return_features:
            return pred
        else:
            return feat, pred

def input_diversity(x, resize_rate=1.10, diversity_prob=0.3):
    if torch.rand(1) < diversity_prob:
        img_size = x.shape[-1]
        img_resize = int(img_size * resize_rate)
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        padded = F.interpolate(padded, size=[img_size, img_size])
        return padded
    else:
        return x



def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()

    cfg.TRAINER.COCOOP = CN()

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # cfg.DATASET.NUM_SHOTS = 16
    cfg.device = args.device


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.bs:
        cfg.DATALOADER.BS = args.bs

    if args.ratio:
        cfg.ratio = args.ratio


def sample_data_by_ratio(data_source, ratio):
    label_to_items = {}
    for item in data_source:
        if item.label not in label_to_items:
            label_to_items[item.label] = []
        label_to_items[item.label].append(item)

    new_data_source = []

    for label, items in label_to_items.items():
        num_items_to_sample = max(1, int(len(items) * ratio))  # 至少抽取一个
        sampled_items = random.sample(items, num_items_to_sample)
        new_data_source.extend(sampled_items)

    return new_data_source