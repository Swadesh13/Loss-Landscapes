import argparse
import os
import time
import torch
from datasets import get_cifar10, get_cifar100, get_celeba
from models import resnet18, resnet34, resnet50
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50"])
parser.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "cifar100", "celeba"])
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                    else "cpu", choices=["cpu", "cuda"])
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--add_norm", action="store_true")
parser.add_argument("--add_skip", action="store_true")
parser.add_argument("--loss", type=str,
                    default="crossentropy", choices=["crossentropy"])
parser.add_argument("--optimizer", type=str, default="adam",
                    choices=["adam", "sgd", "rmsprop"])
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--ckpt_path", type=str, default="ckpt")
args = parser.parse_args()
device = torch.device(args.device)
args.ckpt_path = os.path.join(args.ckpt_path, str(int(time.time())))
print("Args:", args)
os.makedirs(args.ckpt_path)

if args.dataset == "cifar10":
    train_loader, test_loader = get_cifar10(args.image_size, args.batch_size, args.num_workers)
    args.num_classes = 10
elif args.dataset == "cifar100":
    train_loader, test_loader = get_cifar100(args.image_size, args.batch_size, args.num_workers)
    args.num_classes = 100
elif args.dataset == "celeba":
    train_loader, test_loader = get_celeba(args.image_size, args.batch_size, args.num_workers)
    args.num_classes = 40


if args.model == "resnet18":
    model = resnet18(add_norm=args.add_norm, add_skip=args.add_skip,
                     num_classes=args.num_classes).to(device)
elif args.model == "resnet34":
    model = resnet34(add_norm=args.add_norm, add_skip=args.add_skip,
                     num_classes=args.num_classes).to(device)
elif args.model == "resnet50":
    model = resnet50(add_norm=args.add_norm, add_skip=args.add_skip,
                     num_classes=args.num_classes).to(device)

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


for epoch in range(args.epochs):  # loop over the dataset multiple times
    model = model.train()
    running_loss = 0.0
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

    print('[%d] Train loss: %.3f' %
          (epoch + 1, running_loss / len(train_loader)))

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

    print('[%d] Test loss: %.3f' %
          (epoch + 1, running_loss / len(test_loader)), '\n')

    if (epoch+1) % 5 == 0:
        torch.save(model, os.path.join(args.ckpt_path, f"model-{epoch}.pt"))
        print("Model saved")

print('Finished Training')
