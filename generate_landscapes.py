import os
import time
import argparse
import torch
from datasets import get_cifar10, get_cifar100, get_celeba
from models import resnet18, resnet34, resnet50
from loss_landscapes.landscapes import create_2D_losscape, create_3D_losscape

parser = argparse.ArgumentParser()
parser.add_argument("model_ckpt", type=str)
parser.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100"])
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_batches", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--load_once", action="store_true")
parser.add_argument("--loss", type=str,
                    default="crossentropy", choices=["crossentropy"])
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                    else "cpu", choices=["cpu", "cuda"])
parser.add_argument("--output", type=str, default="results")
parser.add_argument("--test", action="store_true")
parser.add_argument("--show_2d", action="store_true")
parser.add_argument("--xmin", type=float, default=-1)
parser.add_argument("--xmax", type=float, default=1)
parser.add_argument("--ymin", type=float, default=-1)
parser.add_argument("--ymax", type=float, default=1)
parser.add_argument("--num_points", type=int, default=50)
parser.add_argument("--show", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--output_vtp", action="store_true")
parser.add_argument("--output_h5", action="store_true")
args = parser.parse_args()
args.output = os.path.join(args.output, str(int(time.time())))
print("Args:", args)
device = torch.device(args.device)


if args.dataset == "cifar10":
    train_loader, test_loader = get_cifar10(args.batch_size, args.num_workers)
    args.num_classes = 10
elif args.dataset == "cifar100":
    train_loader, test_loader = get_cifar100(args.batch_size, args.num_workers)
    args.num_classes = 100
elif args.dataset == "celeba":
    train_loader, test_loader = get_celeba(args.image_size, args.batch_size, args.num_workers)
    args.num_classes = 40

model = torch.load(args.model_ckpt, map_location=device)
if args.loss == "crossentropy":
    if args.dataset == "celeba":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

landscape_func = create_2D_losscape if args.show_2d else create_3D_losscape

landscape_func(
    model=model,
    device=device,
    train_loader_unshuffled=test_loader if args.test else train_loader,
    criterion=criterion,
    num_batches=args.num_batches,
    load_once=args.load_once,
    x_min = args.xmin,
    x_max = args.xmax,
    y_min = args.ymin,
    y_max = args.ymax,
    num_points = args.num_points,
    show = args.show,
    save = args.save,
    output_path = args.output,
    output_vtp = args.output_vtp,
    output_h5 = args.output_h5,
)
