import os
import time
import argparse
import glob
import torch
from models.resnet import resnet18, resnet34, resnet50
from models.nn import simple_nn, nn_bn, conv_nn
from models.cifar_resnet import (
    cifar_resnet20,
    cifar_resnet32,
    cifar_resnet44,
    cifar_resnet56,
    cifar_resnet110,
    cifar_resnet1202,
)
from datasets import get_cifar10, get_cifar100, get_celeba, get_mnist, get_svhn, get_breast_cancer
from loss_landscapes.landscapes import create_2D_losscape, create_3D_losscape

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--ckpt", type=str)
parser.add_argument("--add_norm", action="store_true")
parser.add_argument("--add_skip", action="store_true")
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    choices=["cifar10", "cifar100", "celeba", "mnist", "svhn", "bc", "breast_cancer"],
)
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_batches", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--load_once", action="store_true", help="Load data once")
parser.add_argument("--loss", type=str, default="crossentropy", choices=["crossentropy"])
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    choices=["cpu", "cuda"],
)
parser.add_argument("--output", type=str, default="results")
parser.add_argument("--test", action="store_true")
parser.add_argument("--show_2d", action="store_true")
parser.add_argument("--direction", nargs="*", help="Direction files")
parser.add_argument("--xmin", type=float, default=-1)
parser.add_argument("--xmax", type=float, default=1)
parser.add_argument("--ymin", type=float, default=-1)
parser.add_argument("--ymax", type=float, default=1)
parser.add_argument("--num_points", type=int, default=50)
parser.add_argument("--show", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--output_vtp", action="store_true")
parser.add_argument("--output_h5", action="store_true")
parser.add_argument("--pca", type=str, help="if pca, then provide --pca path/to/weights/dir")
parser.add_argument("--pca_min", type=int, help="if pca, minimum epoch number to consider", default=0)
parser.add_argument("--pca_max", type=int, help="if pca, maximum epoch number to consider", default=1000)
parser.add_argument("--h5", type=str, help="load from existing h5 and create visualizations")
args = parser.parse_args()
args.output = os.path.join(args.output, str(int(time.time())))
os.makedirs(args.output)
print("Args:", args)
device = torch.device(args.device)
with open(os.path.join(args.output, "args.txt"), "w") as f:
    f.write(str(args))

args.channels = 3

if args.dataset == "cifar10":
    train_loader, test_loader = get_cifar10(args.image_size, args.batch_size, args.num_workers, train=True)
    args.num_classes = 10
elif args.dataset == "cifar100":
    train_loader, test_loader = get_cifar100(args.image_size, args.batch_size, args.num_workers, train=True)
    args.num_classes = 100
elif args.dataset == "celeba":
    train_loader, test_loader = get_celeba(args.image_size, args.batch_size, args.num_workers)
    args.num_classes = 40
elif args.dataset == "mnist":
    args.image_size = 28
    train_loader, test_loader = get_mnist(args.batch_size, args.num_workers)
    args.num_classes = 10
    args.channels = 1
elif args.dataset == "svhn":
    args.image_size = 32
    train_loader, test_loader = get_svhn(args.batch_size, args.num_workers, train=True)
    args.num_classes = 10
elif args.dataset in ["bc", "breast_cancer"]:
    train_loader, test_loader = get_breast_cancer()
    train_loader = [train_loader]
    test_loader = [test_loader]
    args.num_classes = 1


if args.model == "resnet20":
    model = cifar_resnet20(args.num_classes, add_norm=args.add_norm, add_skip=args.add_skip)
elif args.model == "resnet32":
    model = cifar_resnet32(args.num_classes, add_norm=args.add_norm, add_skip=args.add_skip)
elif args.model == "resnet44":
    model = cifar_resnet44(args.num_classes, add_norm=args.add_norm, add_skip=args.add_skip)
elif args.model == "resnet56":
    model = cifar_resnet56(args.num_classes, add_norm=args.add_norm, add_skip=args.add_skip)
elif args.model == "resnet110":
    model = cifar_resnet110(args.num_classes, add_norm=args.add_norm, add_skip=args.add_skip)
elif args.model == "resnet1202":
    model = cifar_resnet1202(args.num_classes, add_norm=args.add_norm, add_skip=args.add_skip)
elif args.model == "resnet18":
    model = resnet18(add_norm=args.add_norm, add_skip=args.add_skip, num_classes=args.num_classes)
elif args.model == "resnet34":
    model = resnet34(add_norm=args.add_norm, add_skip=args.add_skip, num_classes=args.num_classes)
elif args.model == "resnet50":
    model = resnet50(add_norm=args.add_norm, add_skip=args.add_skip, num_classes=args.num_classes)
elif args.model == "nn":
    if args.dataset in ["bc", "breast_cancer"]:
        in_dim = 30
    else:
        in_dim = args.image_size * args.image_size * args.channels
    hidden_dim = 256
    model = simple_nn(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=args.num_classes, num_hidden_layers=2)
elif args.model == "nn_bn":
    if args.dataset in ["bc", "breast_cancer"]:
        in_dim = 30
    else:
        in_dim = args.image_size * args.image_size * args.channels
    hidden_dim = 256
    model = nn_bn(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=args.num_classes, num_hidden_layers=10)
elif args.model == "conv_nn":
    model = conv_nn(args.num_classes)

if args.h5:
    if args.show_2d:
        raise NotImplementedError("No h5 support for 2d!")
    else:
        if args.pca:
            weights = glob.glob(os.path.join(args.pca, "*.pt"))
            weights.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
            weights = list(
                filter(lambda x: args.pca_min < int(x.split("-")[-1].split(".")[0]) < args.pca_max, weights)
            )
            model.load_state_dict(torch.load(weights[-1], map_location=device))
            model.eval()
        else:
            model.load_state_dict(torch.load(args.ckpt, map_location=device))
            model.eval()
        create_3D_losscape(
            model=model,
            h5=args.h5,
            pca=weights,
            output_path=args.output,
        )
else:
    if args.loss == "crossentropy":
        if args.dataset == "celeba":
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

    if not args.pca:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()

    if args.pca:
        weights = glob.glob(os.path.join(args.pca, "*.pt"))
        weights.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        weights = list(
            filter(lambda x: args.pca_min <= int(x.split("-")[-1].split(".")[0]) <= args.pca_max, weights)
        )
        model.load_state_dict(torch.load(weights[-1], map_location=device))
        model.eval()
        kwarg = {
            "pca": weights,
            "y_min": args.ymin,
            "y_max": args.ymax,
            "output_vtp": args.output_vtp,
            "output_h5": args.output_h5,
        }
    elif args.direction:
        if not args.show_2d:
            assert len(args.direction) == 2, "Require 2 directions"
        directions = []
        for dir in args.direction[:2]:
            m = torch.load(dir, map_location=device)
            directions.append([p.data for p in m.parameters()])
    else:
        directions = None

    if args.show_2d:
        landscape_func = create_2D_losscape
        kwarg = {"direction": directions}
    else:
        landscape_func = create_3D_losscape
        if not args.pca:
            kwarg = {
                "directions": directions,
                "y_min": args.ymin,
                "y_max": args.ymax,
                "output_vtp": args.output_vtp,
                "output_h5": args.output_h5,
            }

    landscape_func(
        model=model,
        device=device,
        train_loader_unshuffled=test_loader if args.test else train_loader,
        criterion=criterion,
        num_batches=args.num_batches,
        load_once=args.load_once,
        x_min=args.xmin,
        x_max=args.xmax,
        num_points=args.num_points,
        show=args.show,
        save=args.save,
        output_path=args.output,
        **kwarg,
    )
