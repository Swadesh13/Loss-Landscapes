import argparse
import os
import time
import torch
from datasets import get_cifar10, get_cifar100, get_celeba
from models.resnet import resnet18, resnet34, resnet50
from models.nn import simple_nn
from models.cifar_resnet import cifar_resnet20, cifar_resnet32, cifar_resnet44, cifar_resnet56, cifar_resnet110, cifar_resnet1202
from tqdm import tqdm
from torchmetrics import Accuracy
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50", "nn", "resnet20", "resnet32", "resnet44", "resnet56", "resnet110", "resnet1202"])
parser.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100", "celeba"])
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--max_lr_epochs", type=int, default=20)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                    else "cpu", choices=["cpu", "cuda"])
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--add_norm", action="store_true")
parser.add_argument("--add_skip", action="store_true")
parser.add_argument("--loss", type=str,
                    default="crossentropy", choices=["crossentropy"])
parser.add_argument("--optimizer", type=str, default="sgd",
                    choices=["adam", "sgd", "rmsprop"])
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--ckpt_dir", type=str, default="ckpt")
parser.add_argument("--metric", type=str, default="acc", choices=["acc"])
args = parser.parse_args()
device = torch.device(args.device)
args.ckpt_dir = os.path.join(args.ckpt_dir, str(int(time.time())))
print("Args:", args)
os.makedirs(args.ckpt_dir)
with open(os.path.join(args.ckpt_dir, "args.txt"), "w") as f:
    f.write(str(args))
writer = SummaryWriter(args.ckpt_dir)

if args.dataset == "cifar10":
    train_loader, test_loader = get_cifar10(
        args.image_size, args.batch_size, args.num_workers)
    args.num_classes = 10
elif args.dataset == "cifar100":
    train_loader, test_loader = get_cifar100(
        args.image_size, args.batch_size, args.num_workers)
    args.num_classes = 100
elif args.dataset == "celeba":
    train_loader, test_loader = get_celeba(
        args.image_size, args.batch_size, args.num_workers)
    args.num_classes = 40

if "cifar" in args.dataset:
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
if args.model == "resnet18":
    model = resnet18(add_norm=args.add_norm, add_skip=args.add_skip,
                     num_classes=args.num_classes)
elif args.model == "resnet34":
    model = resnet34(add_norm=args.add_norm, add_skip=args.add_skip,
                     num_classes=args.num_classes)
elif args.model == "resnet50":
    model = resnet50(add_norm=args.add_norm, add_skip=args.add_skip,
                     num_classes=args.num_classes)
elif args.model == "nn":
    in_dim = args.image_size*args.image_size*3
    hidden_dim = 1024
    model = simple_nn(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=args.num_classes)

model = model.to(device)

if args.loss == "crossentropy":
    if args.dataset == "celeba":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == "rmsprop":
    optimizer = torch.optim.RMSprop(model.parameters(
    ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.metric == "acc":
    if args.dataset == "celeba":
        metric = Accuracy(task="multilabel", num_labels=args.num_classes).to(device)
    else:
        metric = Accuracy(task="multiclass", num_classes=args.num_classes).to(device)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.max_lr_epochs, args.lr*5e-3)

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

    print('[%d] Train loss: %.3f' %
          (epoch + 1, running_loss / len(train_loader)))
    print(f'Train {args.metric}: {metric.compute()}')
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

    print('[%d] Test loss: %.3f' %
          (epoch + 1, running_loss / len(test_loader)))
    print(f'Test {args.metric}: {metric.compute()}', '\n')
    writer.add_scalar("Loss/test", running_loss / len(test_loader), epoch)
    writer.add_scalar("Acc/test", metric.compute(), epoch)
    metric.reset()

    if epoch >= args.max_lr_epochs:
        lr_scheduler.step()
    
    if (epoch+1) % 5 == 0:
        torch.save(model, os.path.join(args.ckpt_dir, f"model-{epoch}.pt"))
        print("Model saved")
    writer.flush()

print('Finished Training')
writer.close()