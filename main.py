import argparse
import torch
import os

from dass.utils import setup_logger, set_random_seed, collect_env_info
from dass.config import get_cfg_default
from dass.engine import build_trainer
import torch.optim as optim
from loss.head.head_def import HeadFactory
import yaml
from ruamel.yaml import YAML
import torchvision.transforms as transforms
from ignite.metrics import Accuracy
import foolbox as fb
import torch.nn as nn
import torch.nn.functional as F
from utils.util import setup_cfg
from torchvision.models import *
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils.util import *
from model import UNetLikeGenerator as UNet

##HL Imports
import matplotlib.pyplot as plt
import math
from llmApiUtils import classify_image_qwen, load_prompts_for_dataset, encode_image_to_base64, parse_list, accuracy_calc_for_llm, oxford_pets_to_binary
import torchvision.transforms.functional as TF
from PIL import Image
import tempfile
from pathlib import Path
import requests
import json
import base64


# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip

class AdversarialTrainer:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.root = args.root
        self.device = args.device
        self.target = args.target
        self.surrogate = args.surrogate
        self.eps = args.eps
        self.surrogate_path = f'{self.root}/{args.dataset}/{args.surrogate}_{args.head}.pth'
        self.target_path = f'{self.root}/{args.dataset}/{args.target}.pt'
        
        ##HL addition for setting openrouter api key
        self.openrouter_api_key = args.apikey
        self.openrouter_headers = {"Authorization": f"Bearer {self.openrouter_api_key}", "Content-Type": "application/json"}

        ##Hl addition for image path save
        self.figure_save_path_root = f'{self.root}/{args.dataset}/{args.surrogate}_trainplot.png'
        if args.adv_training:
            self.target_path = f'{self.root}/{args.dataset}/{args.target}_adv.pt'
        self.adv_path = f'{self.root}/{args.dataset}/{args.surrogate}_{args.head}/{args.attack}.pth'

        self.ensure_dir()
        self.trainer = build_trainer(cfg)
        self.setup_data()
        if args.attack == 'FGSM':
            self.num_iter = 1

        ##HL Addition: toggle for swithcing between crossentropy loss and bcewithlogits loss (for sigmoid-based losses##
        self.use_bcewithlogits = False if args.head.lower() not in ["sigliphead", "arcfacesigmoid"] else True
        ##HL Addition: if using multiclasshingeloss, use this to control which criterion to use + set the margin val
        self.hingeloss_margin = None if args.head.lower() not in ["hingelosshead"] else 1.0

    def ensure_dir(self):
        for file_path in [self.surrogate_path, self.target_path, self.adv_path]:
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

    def setup_surrogate(self):
        yaml = YAML(typ='safe')
        config = yaml.load(open('configs/data.yaml', 'r'))
        config['num_classes'] = self.trainer.dm.num_classes
        config['output_dim'] = 1024
        head_factory = HeadFactory(args.head, config)
        backbone = self.trainer.clip_model.visual
        backbone = self.wrap_model(backbone)
        self.surrogate = Model(backbone, head_factory).to(self.device)
        print(f"model architecture: \n\n {self.surrogate} \n\n")
        print("Trainable parameters:")
        for name, param in self.surrogate.named_parameters():
            if param.requires_grad:
                print(f"{name} | shape: {tuple(param.shape)}")

    def setup_target(self, name='rn18'):
        num_classes = self.trainer.dm.num_classes
        if name == 'rn18':
            model = resnet18(num_classes=num_classes)
        elif name == 'eff':
            model = efficientnet_b0(num_classes=num_classes)
        elif name == 'regnet':
            model = regnet_x_1_6gf(num_classes=num_classes)
        else:
            raise NameError
        self.target = self.wrap_model(model).to(self.device)

    def setup_data(self):
        self.train_loader = self.trainer.train_loader_x
        self.test_loader = self.trainer.test_loader
        self.mf_loader = self.trainer.mf_loader
        self.sub_mf_loader = self.trainer.sub_mf_loader
        self.sub_test_loader = self.trainer.sub_test_loader
        print('dataset loaded')

    def setup_optimization(self, model, num_epoch, lr=3e-4, optimizer='AdamW'):
        if optimizer == 'AdamW':
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        else:
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     self.args.num_epoch * len(loader),
        #     eta_min=1e-6
        # )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=int(num_epoch/2), T_mult=1)
        if self.use_bcewithlogits:
            print("Using BCEWithLogitsLoss for Sigmoid-based head.")
            self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        elif self.hingeloss_margin is not None:
            print(f"Using MultiClassHingeLoss for Hinge-loss head with margin {self.hingeloss_margin}.")
            self.criterion = nn.MultiMarginLoss(margin=self.hingeloss_margin).to(self.device)
        else:
            print("Using CrossEntropyLoss for Softmax-based head.")
            self.criterion = nn.CrossEntropyLoss().to(self.device)

    def wrap_model(self, model):
        normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        return torch.nn.Sequential(
            normalize,
            model
        )

    def finetune(self, num_epoch):
        #HL Addition: dict for holding train epoch loss/acc
        self.train_acc_loss = {}
        loader = self.mf_loader

        ##TODO: also implement resume training for optimizer and scheduler states
        ##HL Addition: need to implement resume training logic since authors have this flag set but no implementation##
        if args.resume and len(args.resume)>0 and os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            self.load_model(self.surrogate, args.resume)
        ##End##
        
        self.setup_optimization(model=self.surrogate, num_epoch=num_epoch, optimizer=args.optimizer, lr=args.lr)
        for epoch in range(num_epoch):
            train_acc = Accuracy()
            train_acc, loss = self.train_one_epoch(self.surrogate, train_acc, loader)
            self.train_acc_loss[epoch] = [train_acc.compute(), loss]
            print(f'Epoch: {epoch}, train acc: {train_acc.compute():.4f}, loss: {loss:.6f}')

        self.graph_train_loss_and_acc(self.train_acc_loss, self.figure_save_path_root)
        self.eval_one_epoch(self.surrogate, self.test_loader)


    ##HL addition: function for graphing train loss and acc graphs
    def graph_train_loss_and_acc(self, train_log: dict, out_path:str):
        train_loss = [a[1] for a in train_log.values()]
        train_acc = [a[0] for a in train_log.values()]
        plt.figure(figsize=(8, 5))
        plt.plot(train_log.keys(), train_acc, label=f"Train Accuracy")
        plt.plot(train_log.keys(), train_loss, label=f"Train Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Training Accuracy and Loss")
        plt.title("Training Accuracy and Loss over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_path)
        
    def train_scratch(self, num_epoch=90,):
        self.setup_target(self.target)
        loader = self.train_loader
        self.setup_optimization(model=self.target, num_epoch=num_epoch, optimizer='SGD', lr=1e-1)
        for epoch in range(num_epoch):
            train_acc = Accuracy()
            train_acc, loss = self.train_one_epoch(self.target, train_acc, loader)
            print(f'Epoch: {epoch}, train acc: {train_acc.compute():.4f}, loss: {loss:.6f}')
        self.eval_one_epoch(self.target, self.test_loader)

    def train_one_epoch(self, model, train_acc, loader):
        model.train()
        total_loss = 0
        for batch in loader:
            images = batch['img'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            if args.adv_training:
                images = self.pgd_attack(model, images, labels, epsilon=self.eps / 255., num_iter=self.num_iter)
            else:
                images = images
            try:
                outputs = model(images, labels)
            except TypeError:
                outputs = model(images)

            ##HL addition: need to one-hot encode labels for bcewithlogits loss##

            if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
                # Convert integer labels → one-hot for BCE
                labels_onehot = F.one_hot(labels, num_classes=outputs.size(1)).float()
                loss = self.criterion(outputs, labels_onehot)
            else:
                #hingeloss and crossentropy can use interger labels
                loss = self.criterion(outputs, labels)

            #old code:
            # loss = self.criterion(outputs, labels)

            ##end HL addition##

            loss.backward()
            self.optimizer.step()

            train_acc.update((outputs, labels))
            total_loss += loss.item()
        self.scheduler.step()

        return train_acc, total_loss / len(loader)

    def eval_one_epoch(self, model, loader):
        model.eval()
        acc = Accuracy()
        for batch in loader:
            images = batch['img'].to(self.device)
            labels = batch['label'].to(self.device)

            with torch.no_grad():
                try:
                    outputs = model(images, labels)
                except TypeError:
                    outputs = model(images)
                acc.update((outputs, labels))
        print('eval acc: {:.4f}'.format(acc.compute()))
        return acc.compute()
            
    
    ##HL Addition: helper for llm inference
    def llm_predict_batch(self, images_tensor):
        """
        images_tensor: (B, 3, 224, 224) in [0,1]
        Returns: tensor of predicted class indices (B,)
        """

        B = images_tensor.size(0)
        encoded_images = []

        # --- encode batch of images ---
        for img in images_tensor:
            pil_img = TF.to_pil_image(img.cpu())
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                pil_img.save(f.name)
                b64_img = encode_image_to_base64(f.name)
                encoded_images.append(b64_img)

        # Load prompts
        system_prompt, user_prompt = load_prompts_for_dataset(args.dataset)

        # Append the JSON instruction BEFORE adding to content_list
        user_prompt = (
            user_prompt
            + f"\nReturn ONLY a Python list of integers, one per image. For example: [1, 5, 12]. Make sure the number of items in the list matches the {B} images passed to you. Rember to properly open and close the list with [ and ]"
        )

        # Build content list: text + N images
        content_list = [
            {"type": "text", "text": user_prompt},
        ]
        for b64 in encoded_images:
            content_list.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{b64}"
            })

        payload = {
            "model": "qwen/qwen3-vl-30b-a3b-instruct",
            "temperature": 0,
            "max_tokens": 512,        # enough for 128 ints
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": content_list}
            ]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=self.openrouter_headers,
            json=payload
        )

        # Validate response
        try:
            response_json = response.json()
        except:
            print("Invalid JSON from API:", response.text)
            return torch.full((B,), -1, dtype=torch.long, device=self.device)

        # Extract output
        try:
            raw_text = response_json["choices"][0]["message"]["content"].strip()
        except:
            print("Malformed response:", response_json)
            return torch.full((B,), -1, dtype=torch.long, device=self.device)

        preds = parse_list(raw_text, B)

        # print(f"Raw text: {raw_text}")
        # print(f"Parsed list: {preds}")

        # Convert to tensor
        preds = torch.tensor(preds, dtype=torch.long, device=self.device)

        # Pad or trim to batch size
        if preds.size(0) < B:
            pad = torch.zeros((B - preds.size(0),), dtype=torch.long, device=self.device)
            preds = torch.cat([preds, pad], dim=0)
        elif preds.size(0) > B:
            preds = preds[:B]

        # Clamp invalid indices
        num_classes = self.trainer.dm.num_classes
        preds = torch.clamp(preds, 0, num_classes - 1)


        return preds

    
    def eval_adv(self, batch_size):

        # -----------------------------
        # Apply rate limit
        # -----------------------------
        limit = self.args.inferlimit
        full_dataset_size = len(self.test_loader.dataset)

        if limit is not None:
            limit = min(limit, full_dataset_size)
            print(f"[Rate Limit] Evaluating only first {limit} images out of {full_dataset_size}")
        else:
            limit = full_dataset_size

        # -----------------------------
        # Load UNet attack
        # -----------------------------
        unet = UNet().to(self.device)
        ckpt = torch.load(f'{self.root}/{self.args.dataset}/unet.pt', map_location='cpu')
        unet.load_state_dict(ckpt)
        unet.eval()

        loader = self.test_loader

        # -----------------------------
        # Pre-allocate tensors
        # -----------------------------
        adv_examples = torch.empty(size=[limit, 3, 224, 224])
        adv_labels   = torch.empty(size=[limit], dtype=torch.long)

        total_gt_labels_fine = []
        total_gt_labels_binary = []

        # -----------------------------
        # Generate adversarial examples
        # -----------------------------
        filled = 0
        for batch in loader:
            images = batch['img'].to(self.device)
            labels = batch['label'].to(self.device)

            total_gt_labels_fine.extend(labels.cpu().tolist())

            if self.args.dataset == "oxford_pets":
                total_gt_labels_binary.extend([oxford_pets_to_binary(x) for x in labels.cpu().tolist()])

            bsz = images.size(0)
            if filled + bsz > limit:
                bsz = limit - filled
                images = images[:bsz]
                labels = labels[:bsz]

            with torch.no_grad():
                noise = unet(images)
                noise = torch.clamp(noise, -self.eps/255., self.eps/255.)
                images_adv = torch.clamp(images + noise, 0, 1)

            adv_examples[filled:filled+bsz] = images_adv.cpu()
            adv_labels[filled:filled+bsz]   = labels.cpu()

            filled += bsz
            if filled >= limit:
                break

        print("GT fine labels:", total_gt_labels_fine)
        if self.args.dataset == "oxford_pets":
            print("GT binary labels:", total_gt_labels_binary)

        # -----------------------------
        # Evaluate each target model
        # -----------------------------
        targets = ["rn18", "eff", "regnet", "qwen_api"]

        for target in targets:

            print("\n=========================================")
            print(f"Evaluating target = {target}")

            is_api = ("_api" in target)
            is_pets = (self.args.dataset == "oxford_pets")

            # Load local model if needed
            if not is_api:
                self.setup_target(name=target)
                self.load_model(self.target, f'{self.root}/{self.args.dataset}/{target}.pt')
                model = self.target.eval().cuda()

            # storage
            adv_preds_fine = []
            adv_preds_binary = []

            clean_preds_fine = []
            clean_preds_binary = []

            num_batches = (limit + batch_size - 1) // batch_size

            # ---------------------- ADV ACC ----------------------
            acc_local_adv = Accuracy()

            for batch_idx in range(num_batches):

                si = batch_idx * batch_size
                ei = min(si + batch_size, limit)

                imgs = adv_examples[si:ei].to(self.device)
                labels = adv_labels[si:ei].to(self.device)

                if is_api:
                    preds_fine = self.llm_predict_batch(imgs).detach().cpu().tolist()
                    adv_preds_fine.extend(preds_fine)

                    if is_pets:
                        preds_binary = [oxford_pets_to_binary(x) for x in preds_fine]
                        adv_preds_binary.extend(preds_binary)

                else:
                    with torch.no_grad():
                        logits = model(imgs)
                        acc_local_adv.update((logits, labels))

                        if is_pets:
                            preds_fine_local = torch.argmax(logits, dim=1).cpu().tolist()
                            adv_preds_fine.extend(preds_fine_local)

                            adv_preds_binary.extend([oxford_pets_to_binary(x) for x in preds_fine_local])
            
            # Fix reversed order only for pets API
            if is_pets and is_api:
                adv_preds_fine = list(reversed(adv_preds_fine))
                adv_preds_binary = list(reversed(adv_preds_binary))

            # Final adv accuracy
            if is_api:
                adv_acc_fine = accuracy_calc_for_llm(adv_preds_fine, total_gt_labels_fine)
            else:
                adv_acc_fine = acc_local_adv.compute()

            if is_pets:
                adv_acc_binary = accuracy_calc_for_llm(adv_preds_binary, total_gt_labels_binary)

            # ---------------------- CLEAN ACC ----------------------
            acc_local_clean = Accuracy()

            filled = 0
            for batch in loader:

                imgs = batch['img'].cuda()
                labels = batch['label'].cuda()
                bsz = imgs.size(0)

                if filled + bsz > limit:
                    bsz = limit - filled
                    imgs = imgs[:bsz]
                    labels = labels[:bsz]

                if is_api:
                    preds_fine = self.llm_predict_batch(imgs).detach().cpu().tolist()
                    clean_preds_fine.extend(preds_fine)

                    if is_pets:
                        clean_preds_binary.extend([oxford_pets_to_binary(x) for x in preds_fine])

                else:
                    with torch.no_grad():
                        logits = model(imgs)
                        acc_local_clean.update((logits, labels))

                        if is_pets:
                            preds_fine_local = torch.argmax(logits, dim=1).cpu().tolist()
                            clean_preds_fine.extend(preds_fine_local)
                            clean_preds_binary.extend([oxford_pets_to_binary(x) for x in preds_fine_local])

                filled += bsz
                if filled >= limit:
                    break

            # Fix order only for pets API
            if is_pets and is_api:
                clean_preds_fine = list(reversed(clean_preds_fine))
                clean_preds_binary = list(reversed(clean_preds_binary))

            # clean accuracy
            if is_api:
                clean_acc_fine = accuracy_calc_for_llm(clean_preds_fine, total_gt_labels_fine)
            else:
                clean_acc_fine = acc_local_clean.compute()

            if is_pets:
                clean_acc_binary = accuracy_calc_for_llm(clean_preds_binary, total_gt_labels_binary)

            # -----------------------------
            # Print results
            # -----------------------------
            print(f"[Fine 37-class]  ASR: {clean_acc_fine - adv_acc_fine:.4f}, clean: {clean_acc_fine:.4f}, adv: {adv_acc_fine:.4f}")

            if is_pets:
                print(f"[Binary cat/dog] ASR: {clean_acc_binary - adv_acc_binary:.4f}, clean: {clean_acc_binary:.4f}, adv: {adv_acc_binary:.4f}")




    def save_model(self, model, ckpts):
        torch.save(model.state_dict(), ckpts)

    def load_model(self, model, ckpts):
        model.load_state_dict(torch.load(ckpts, map_location='cpu'))

    def pgd_attack(self, model, images, labels, epsilon=0.03, num_iter=10, random_start=True):
        alpha = (epsilon / num_iter)
        orig_images = images.clone().detach()

        if random_start:
            noise = torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(self.device)
            images = torch.clamp(images + noise, min=0, max=1).detach()

        for _ in range(num_iter):
            images.requires_grad = True
            try:
                outputs = model(images, labels)
            except TypeError:
                outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            model.zero_grad()
            loss.backward()

            adv_images = images + alpha * images.grad.sign()
            eta = torch.clamp(adv_images - orig_images, min=-epsilon, max=epsilon)
            images = torch.clamp(orig_images + eta, min=0, max=1).detach()

        return images

    def train_unet(self, num_epoch=90):
        print(f"unet will be saved as: {self.root}/{args.dataset}/{args.head}_unet.pt")
        unet = UNet().to(self.device)
        loader = self.mf_loader
        optimizer = optim.AdamW(unet.parameters(), lr=1e-4)
        self.load_model(model=self.surrogate, ckpts=self.surrogate_path)
        unet.train()
        criterion = nn.CrossEntropyLoss().to(self.device)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(num_epoch / 2), T_mult=1)
        self.surrogate.eval().to(self.device)
        for epoch in range(num_epoch):
            train_acc = Accuracy()
            total_loss = 0
            for batch in loader:
                images = batch['img'].to(self.device)
                labels = batch['label'].to(self.device)
                optimizer.zero_grad()

                noise = unet(images)
                noise = torch.clamp(noise, -self.eps/255., self.eps/255.)
                images_adv = images + noise
                images_adv = torch.clamp(images_adv, 0, 1)
                outputs = self.surrogate(images_adv, labels)

                loss = 10 - criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_acc.update((outputs, labels))
                total_loss += loss.item()
            scheduler.step()

            # return train_acc, total_loss / len(loader)
            print(f'Epoch: {epoch}, train acc: {train_acc.compute():.4f}, loss: {total_loss / len(loader):.6f}')
        self.save_model(unet, f'{self.root}/{args.dataset}/{args.head}_unet.pt')

    def run(self):
        if self.args.flag == 'finetune':
            self.setup_surrogate()
            self.finetune(num_epoch=args.num_epoch)
            self.save_model(model=self.surrogate, ckpts=self.surrogate_path)
        elif self.args.flag == 'train_scratch':
            self.train_scratch(num_epoch=args.num_epoch)
            self.save_model(self.target, self.target_path)
            print('saved target model')
        elif self.args.flag == 'train_unet':
            self.setup_surrogate()
            self.train_unet(num_epoch=args.num_epoch)
        elif self.args.flag == 'eval_adv':
            # self.eval_adv(batch_size=512)
            ##HL mod: DO NOT USE 512 BS SINCE WE USING LLM APIS
            self.eval_adv(batch_size=args.bs)
        else:
            raise NameError



def main(args):
    cfg = setup_cfg(args)
    trainer = AdversarialTrainer(cfg, args)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="path to dataset")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/CoOp/rn50.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/oxford_pets.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="ZeroshotCLIP", help="name of trainer")
    parser.add_argument("--surrogate", type=str, default="RN50", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="ArcFace", help="name of head")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--flag", type=str, default="train_scratch")
    parser.add_argument("--num_epoch", type=int, default=120)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--eps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--target", type=str, default='rn18')
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument("--dataset", type=str, default='oxford_pets')
    parser.add_argument("--attack", type=str, default='FGSM')
    parser.add_argument("--adv_training", action="store_true")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )

    #HL addition for openrouter api key arg
    parser.add_argument(
        "--apikey",
        default=None,
        type=str
    )
    ##HL addition: added limiter for inference
    parser.add_argument("--inferlimit", type=int, default=None,
                    help="Limit number of images for UNet + LLM evaluation")

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )

    args = parser.parse_args()
    main(args)
