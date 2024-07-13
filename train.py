"""Training procedure for gan.
"""

import os
import argparse
from random import randrange
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import (
    RandomApply,
    RandomCrop,
    RandomHorizontalFlip,
    ColorJitter,
)
from torch.utils.data import DataLoader
from torch.optim import Adam
from bigGAN import BigGAN


def train(
    gan: BigGAN,
    trainloader: DataLoader,
    optimizer_d: Adam,
    optimizer_g: Adam,
    epoch: int,
):
    gan.train()  # set to training mode

    total_loss_d = 0
    total_loss_g = 0
    batch_idx = 0
    for batch, labels in trainloader:
        data = batch.to(gan.device)
        batch_size = data.size()[0]
        labels = labels.to(gan.device)

        # Discriminator training
        optimizer_d.zero_grad()
        fake_images, fake_labels = gan.generate_fake(batch_size)
        loss_d = gan.calculate_discriminator_loss(
            batch, labels, fake_images.detach(), fake_labels
        )
        loss_d.backward()
        optimizer_d.step()

        # Generator training
        optimizer_g.zero_grad()
        fake_images, fake_labels = gan.generate_fake(batch_size)
        loss_g = gan.calculate_generator_loss(fake_images, fake_labels)
        loss_g.backward()
        optimizer_g.step()

        total_loss_d += loss_d.item()
        total_loss_g += loss_g.item()
        batch_idx += 1

    return total_loss_d / batch_idx, total_loss_g / batch_idx


def test(
    gan: BigGAN,
    testloader: DataLoader,
    filename: str,
    epoch: int,
    fixed_noise,
    fixed_labels,
):
    gan.eval()
    with torch.no_grad():
        samples = gan.generator(fixed_noise, fixed_labels)
        torchvision.utils.save_image(
            torchvision.utils.make_grid(samples),
            "./samples/" + filename + f"epoch{epoch}.png",
        )

        total_loss_g = 0
        total_loss_d = 0
        batch_idx = 0
        for batch, labels in testloader:
            batch = batch.to(gan.device)
            labels = labels.to(gan.device)
            batch_size = batch.size(0)

            fake_images, fake_labels = gan.generate_fake(batch_size)
            loss_d = gan.calculate_discriminator_loss(
                batch, labels, fake_images, fake_labels
            )
            loss_g = gan.calculate_generator_loss(fake_images, fake_labels)

            total_loss_d += loss_d.item()
            total_loss_g += loss_g.item()
            batch_idx += 1

        print(
            f"Epoch: {epoch} Test set: Average loss_d: {total_loss_d / batch_idx:.4f}"
        )
        print(
            f"Epoch: {epoch} Test set: Average loss_g: {total_loss_g / batch_idx:.4f}"
        )
    return total_loss_d / batch_idx, total_loss_g / batch_idx


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mnist":
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (32),
                    interpolation=transforms.InterpolationMode.BICUBIC,  # size_that_worked = 64
                ),
                transforms.Grayscale(num_output_channels=3),  # Convert to RGB
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        trainset = torchvision.datasets.MNIST(
            root="./data/MNIST", train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.MNIST(
            root="./data/MNIST", train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
    elif args.dataset == "cifar":
        transform = transforms.Compose(
            [
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                RandomApply(
                    [
                        ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root="./data/Cifar10",
            train=True,
            download=True,
            transform=transform,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data/Cifar10", download=True, transform=transform, train=False
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
    elif args.dataset == "celeba":
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = torchvision.datasets.CelebA(
            root="./data/CelebA",
            split="train",
            download=True,
            transform=transform,
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        testset = torchvision.datasets.CelebA(
            root="./data/CelebA",
            download=True,
            transform=transform,
            split="test",
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

    else:
        raise ValueError("Dataset not implemented")

    filename = (
        "%s_" % args.dataset + "batch%d_" % args.batch_size + "mid%d_" % args.latent_dim
    )

    biggan = BigGAN(
        latent_dim=args.latent_dim,
        num_classes=10,
        img_size=32,
        img_channels=3,
        device=device,
    )
    optimizer_d = torch.optim.Adam(
        biggan.discriminator.parameters(), lr=args.lr * 3, betas=(0.5, 0.999)
    )
    optimizer_g = torch.optim.Adam(
        biggan.generator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=30, gamma=0.5)
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=30, gamma=0.5)
    loss_train_arr_d = []
    loss_test_arr_d = []
    loss_train_arr_g = []
    loss_test_arr_g = []

    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    fixed_labels = (
        torch.arange(10, device=device).repeat(6).long().to(device)
    )  # 0-9 repeated 6 times, plus 4 random
    for epoch in range(1, args.epochs + 1):
        loss_train_d, loss_train_g = train(
            biggan,
            trainloader,
            optimizer_d=optimizer_d,
            optimizer_g=optimizer_g,
            epoch=epoch,
        )
        loss_train_arr_d.append(loss_train_d)
        loss_train_arr_g.append(loss_train_g)
        loss_test_d, loss_test_g = test(
            biggan, testloader, filename, epoch, fixed_noise, fixed_labels
        )
        loss_test_arr_d.append(loss_test_d)
        loss_test_arr_g.append(loss_test_g)

        # Step the learning rate schedulers
        scheduler_d.step()
        scheduler_g.step()

        if epoch % 10 == 0 and epoch != 0:
            # Save the model
            torch.save(biggan.generator.state_dict(), "generator.pt")
            torch.save(biggan.discriminator.state_dict(), "discriminator.pt")
            # create a plot of the loss
            plt.plot(loss_train_arr_d, label="train_d")
            plt.plot(loss_test_arr_d, label="test_d")
            plt.plot(loss_train_arr_g, label="train_g")
            plt.plot(loss_test_arr_g, label="test_g")
            plt.xlabel("Epoch")
            plt.ylabel("gan loss")
            plt.legend()
            plt.savefig("loss.png")
            # reset the plot completely
            plt.clf()
            plt.cla()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--dataset", help="dataset to be modeled.", type=str, default="mnist"
    )
    parser.add_argument(
        "--batch_size", help="number of images in a mini-batch.", type=int, default=64
    )
    parser.add_argument(
        "--epochs", help="maximum number of iterations.", type=int, default=20
    )
    parser.add_argument(
        "--sample_size", help="number of images to generate.", type=int, default=64
    )

    parser.add_argument("--latent-dim", help=".", type=int, default=100)
    parser.add_argument(
        "--lr", help="initial learning rate.", type=float, default=0.0002
    )

    args = parser.parse_args()

    # create samples and models folder iff they don't exist:
    if not os.path.exists("samples"):
        os.makedirs("samples")
    if not os.path.exists("models"):
        os.makedirs("models")
    main(args)
