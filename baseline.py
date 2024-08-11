import argparse
import os
import torch
import transferattack
from transferattack.utils import *
from main import AdversarialTrainer
from utils.util import *
from ignite.metrics import Accuracy


def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=32, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet50', type=str, help='the source surrogate model')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument("--adv_training", action="store_true")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/oxford_pets.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument(
        "--config-file", type=str, default="configs/trainers/CoOp/rn50.yaml", help="path to config file"
    )
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
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument("--trainer", type=str, default="ZeroshotCLIP", help="name of trainer")
    parser.add_argument("--head", type=str, default="ArcFace", help="name of head")
    parser.add_argument("--surrogate", type=str, default="RN50", help="name of CNN backbone")
    parser.add_argument("--root", type=str, help="path to dataset")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument('--GPU_ID', default='0', type=str)
    parser.add_argument("--dataset", type=str, default='oxford_pets')
    parser.add_argument("--target", type=str, default='rn18')
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument('--attack', type=str, help='the attack algorithm',
                        choices=transferattack.attack_zoo.keys())
    return parser.parse_args()


def main():
    args = get_parser()
    args.device = f"cuda:{int(args.GPU_ID)}"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg = setup_cfg(args)
    trainer = AdversarialTrainer(cfg, args)

    targets = ["rn18", "eff", "regnet"]

    dataloader = trainer.test_loader

    print(f'length of dataset is {len(dataloader.dataset)}')

    if args.ensemble or len(args.model.split(',')) > 1:
        args.model = args.model.split(',')
    attacker = transferattack.load_attack_class(args.attack)(model_name=args.model, targeted=args.targeted)

    adv_examples = torch.empty(size=[len(dataloader.dataset), 3, 224, 224])
    adv_labels = torch.empty(size=[len(dataloader.dataset),])
    for batch_idx, batch in enumerate(dataloader):
        images = batch['img'].cuda()
        labels = batch['label'].cuda()
        perturbations = attacker(images, labels)

        noise = torch.clamp(perturbations, -args.eps, args.eps)
        images_adv = images + noise
        images_adv = torch.clamp(images_adv, 0, 1)

        adv_examples[batch_idx * dataloader.batch_size:
                     (batch_idx + 1) * dataloader.batch_size] = images_adv.cpu()
        adv_labels[batch_idx * dataloader.batch_size:
                   (batch_idx + 1) * dataloader.batch_size] = labels.cpu()

    print('start testing')
    del attacker

    for target in targets:
        trainer.setup_target(name=target)
        trainer.load_model(model=trainer.target, ckpts=f'{args.root}/{args.dataset}/{target}.pt')
        model = trainer.target
        model.eval().cuda()
        acc = Accuracy()
        images_array, labels_array = adv_examples, adv_labels
        num_batches = (len(images_array) + dataloader.batch_size - 1) // dataloader.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + dataloader.batch_size, len(images_array))

            images = images_array[start_idx:end_idx].cuda()
            labels = labels_array[start_idx:end_idx].cuda()

            with torch.no_grad():
                outputs = model(images)
                acc.update((outputs, labels))
        # print('eval acc: {:.4f}'.format(acc.compute()))
        adv_acc = acc.compute()
        acc = Accuracy()
        for batch in dataloader:
            images = batch['img'].cuda()
            labels = batch['label'].cuda()

            with torch.no_grad():
                outputs = model(images)
                acc.update((outputs, labels))
        clean_acc = acc.compute()
        print(f'attack:{args.attack}, dataset:{args.dataset}, target:{target}, ASR: {clean_acc - adv_acc:.4f}, clean: {clean_acc:.4f}, adv: {adv_acc:.4f}')
        print('----------------------------------------------------------')



if __name__ == '__main__':
    main()
