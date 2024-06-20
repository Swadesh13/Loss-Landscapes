import argparse
import os
import time
import torch
from datasets import get_cifar10, get_cifar100, get_celeba, get_mnist, get_svhn, get_breast_cancer
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
from tqdm import tqdm
from torchmetrics import Accuracy
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="resnet18",
    choices=[
        "resnet18",
        "resnet34",
        "resnet50",
        "nn",
        "nn_bn",
        "conv_nn",
        "resnet20",
        "resnet32",
        "resnet44",
        "resnet56",
        "resnet110",
        "resnet1202",
    ],
)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    choices=["cifar10", "cifar100", "celeba", "mnist", "svhn", "bc", "breast_cancer"],
)
parser.add_argument("--image_size", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda"]
)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--add_norm", action="store_true")
parser.add_argument("--add_skip", action="store_true")
parser.add_argument("--loss", type=str, default="crossentropy", choices=["crossentropy"])
parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd", "rmsprop"])
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--ckpt_dir", type=str, default="ckpt")
parser.add_argument("--ckpt_step", type=int, default=4)
parser.add_argument("--metric", type=str, default="acc", choices=["acc"])
args = parser.parse_args()
torch.manual_seed(42)
device = torch.device(args.device)
args.ckpt_dir = os.path.join(args.ckpt_dir, str(int(time.time())))
print("Args:", args)
os.makedirs(args.ckpt_dir)
with open(os.path.join(args.ckpt_dir, "args.txt"), "w") as f:
    f.write(str(args))
writer = SummaryWriter(args.ckpt_dir)
args.channels = 3
args.num_classes = None

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
    hidden_dim = 64
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

model = model.to(device)

if args.loss == "crossentropy":
    if args.dataset == "celeba" or args.num_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
elif args.optimizer == "rmsprop":
    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

if args.metric == "acc":
    if args.dataset == "celeba":
        metric = Accuracy(task="multilabel", num_labels=args.num_classes).to(device)
    elif args.num_classes == 1:
        metric = Accuracy(task="binary").to(device)
    else:
        metric = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)

# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, args.epochs - args.max_lr_epochs, args.lr * 1e-3
# )
# lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, args.lr, args.epochs, pct_start=0.05, div_factor=10, final_div_factor=1e3
# )
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 125])


## For skip without norm training
# def warmup_lr_step(epoch):
#     scale = 1
#     if epoch >= 25:
#         scale = 1e3
#     if epoch >= 100:
#         scale *= 0.1
#     if epoch >= 150:
#         scale *= 0.1
#     if epoch >= 175:
#         scale *= 0.1
#     return scale


# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_step)


for epoch in range(args.epochs):  # loop over the dataset multiple times
    model = model.train()
    running_loss = 0.0
    writer.add_scalar("lr", lr_scheduler.get_last_lr()[-1], epoch)
    for i, data in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        metric(outputs, labels)

    print("[%d] Train loss: %.3E" % (epoch + 1, running_loss / len(train_loader)))
    print(f"Train {args.metric}: {metric.compute()}")
    writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)
    writer.add_scalar("Acc/train", metric.compute(), epoch)
    metric.reset()

    model = model.eval()
    running_loss = 0.0
    for i, data in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # print statistics
        running_loss += loss.item()
        metric(outputs, labels)

    print("[%d] Test loss: %.3E" % (epoch + 1, running_loss / len(test_loader)))
    print(f"Test {args.metric}: {metric.compute()}")
    writer.add_scalar("Loss/test", running_loss / len(test_loader), epoch)
    writer.add_scalar("Acc/test", metric.compute(), epoch)
    metric.reset()

    lr_scheduler.step()

    if (epoch + 1) % args.ckpt_step == 0:
        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"model-{epoch+1}.pt"))
        print("Model saved")
    writer.flush()
    print()

print("Finished Training")
writer.close()
