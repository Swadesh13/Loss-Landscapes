import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_cifar10(image_size, batch_size, num_workers=2, train=False):
    if train:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform if train else transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=train, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader


def get_cifar100(image_size, batch_size, num_workers=2, train=False):
    if train:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size, 4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.50707516, 0.48654887, 0.4409178), (0.26733429, 0.25643846, 0.27615047)
                ),
            ]
        )
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.50707516, 0.48654887, 0.4409178), (0.26733429, 0.25643846, 0.27615047)),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform if train else transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=train, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader


def get_mnist(batch_size, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST("./data", True, transform, download=True)
    testset = torchvision.datasets.MNIST("./data", False, transform, download=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return trainloader, testloader


def get_celeba(image_size, batch_size, num_workers=2):
    def convert_float32(x):
        return x.to(torch.float32)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = torchvision.datasets.CelebA(
        root="./data", split="train", download=True, transform=transform, target_transform=convert_float32
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    valset = torchvision.datasets.CelebA(
        root="./data", split="valid", download=True, transform=transform, target_transform=convert_float32
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, valloader


def get_svhn(batch_size, num_workers=2, train=False):
    train_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    if train:
        train_transforms = [transforms.RandomAffine(30, (0.1, 0.1))] + train_transforms
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            root="./data",
            split="train",
            download=True,
            transform=transforms.Compose(train_transforms),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(
            root="./data",
            split="test",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def get_breast_cancer():
    df = pd.read_csv("./data/breast-cancer/data.csv")
    df = df[df.columns[:-1]]
    X = df[df.columns[2:]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X).to(torch.float32)
    y = (df["diagnosis"].values == "M").astype(np.uint8)
    y = torch.tensor(y).to(torch.uint8)
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return (x_train, y_train[:, None].float()), (x_test, y_test[:, None].float())
