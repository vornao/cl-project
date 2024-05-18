import torch
import torch.nn as nn
from utils import get_cifar10_dataloader, get_flowers_dataloader
from models import ConvNetTrainer

# import efficientnet from torchvision models
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)

import argparse

# add argument for tuning only last layer
parser = argparse.ArgumentParser(description="Train a model on CIFAR-10")
parser.add_argument("--lastonly", action="store_true", help="Tune only the last layer")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--model", type=str, default="effnet", help="Model to train")
parser.add_argument(
    "--savepath",
    type=str,
    default="../models/efficientnet_b0.pth",
    help="Path to save the model",
)
args = parser.parse_args()


if __name__ == "__main__":
    # save the model
    # freeze all layers except the last one
    # for param in effnet.parameters():
    #     param.requires_grad = False
    train, valid, test = get_cifar10_dataloader(args.batch_size)
    # resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if args.model == "resnet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif args.model == "effnet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    if args.lastonly:
        for param in model.parameters():
            param.requires_grad = False

    if args.model == "effnet":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    elif args.model == "resnet":
        model.fc = nn.Linear(model.fc.in_features, 10)

    trainer = ConvNetTrainer(
        model=model,
        optimizer="AdamW",
        lr=1e-3,
        weight_decay=1e-6,
        device="mps",
        train_loader=train,
        test_loader=valid,
    )

    baseline_loss, baseline_acc = trainer.test(test)
    print(
        f"Baseline loss: {baseline_loss:.4f}, Baseline accuracy: {baseline_acc*100:.2f}%"
    )
    print(
        f"Training {args.model} model",
        "with last layer only" if args.lastonly else "",
        ", epochs:",
        args.epochs,
    )
    trainer.train(num_epochs=args.epochs)

    # test the model
    test_loss, test_acc = trainer.test(test)
    torch.save(trainer.model.state_dict(), args.savepath)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
