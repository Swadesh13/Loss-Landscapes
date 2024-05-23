import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar10(image_size, batch_size, num_workers=2):
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(
    ), transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def get_cifar100(image_size, batch_size, num_workers=2):
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(
    ), transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def get_celeba(image_size, batch_size, num_workers=2):
    def convert_float32(x):
        return x.to(torch.float32)
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(
    ), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CelebA(
        root='./data', split="train", download=True, transform=transform, target_transform=convert_float32)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valset = torchvision.datasets.CelebA(
        root='./data', split="valid", download=True, transform=transform, target_transform=convert_float32)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, valloader
