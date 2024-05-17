
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt

def plot_mnist(train, width=10, height=5, cmap='gray'): 
    imgs = []
    for i in range(10):
        for img, label in train:
            if label[0] == i:
                imgs.append(img[0])
                break

    #plot the images
    fig, axs = plt.subplots(2, 5, figsize=(width, height))
    for i in range(10):
        ax = axs[i//5, i%5]
        ax.imshow(imgs[i][0], cmap=cmap)
        ax.set_title(f"Class {i}")
        ax.axis('off')

    fig.suptitle("MNIST dataset")
    fig.tight_layout()
    

# load config from../config.json
def load_config():
    import json
    with open("../mnist_config.json", "r") as f:
        config = json.load(f)
    return config


# create MNIST dataloader
def get_mnist_dataloader(batch_size):
    config = load_config()
    # define transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # download and load the MNIST dataset
    train_dataset = datasets.MNIST(
        root=config["datapath"], train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=config["datapath"], train=False, transform=transform
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [50000, 10000]
    )

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_fashion_mnist_dataloader(batch_size):
    config = load_config()
    # define transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # download and load the MNIST dataset
    train_dataset = datasets.FashionMNIST(
        root=config["datapath"], train=True, transform=transform, download=True
    )
    test_dataset = datasets.FashionMNIST(
        root=config["datapath"], train=False, transform=transform
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [50000, 10000]
    )

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_cifar10_dataloader(batch_size):
    config = load_config()
    # define transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # download and load the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root=config["datapath"], train=True, transform=transform, download=True
    )
    test_dataset = datasets.CIFAR10(
        root=config["datapath"], train=False, transform=transform
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [45000, 5000]
    )

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_flowers_dataloader(batch_size):
    config = load_config()
    tr = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # download flowers 102 dataset
    train_dataset = datasets.Flowers102(
        root=config["datapath"], split="train", transform=tr, download=True
    )
    val_dataset = datasets.Flowers102(
        root=config["datapath"], split="val", transform=tr, download=True
    )
    test_dataset = datasets.Flowers102(
        root=config["datapath"], split="test", transform=tr, download=True
    )

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



