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
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from bigGAN import BigGAN


def train(
    gan: BigGAN,
    trainloader: DataLoader,
    optimizer_d: Adam,
    optimizer_g: Adam,
    scaler: GradScaler,
):
    gan.train()
    total_loss_d = 0
    total_loss_g = 0
    batch_idx = 0
    for batch, labels in trainloader:
        batch = batch.to(gan.device)
        batch_size = batch.size()[0]
        labels = labels.to(gan.device)

        # Discriminator training
        for _ in range(2):  # Train discriminator more
            with autocast():
                optimizer_d.zero_grad()
                real_pred = gan.discriminate(batch, labels)
                fake_images, fake_labels = gan.generate_fake(batch_size)
                fake_pred = gan.discriminate(fake_images.detach(), fake_labels)
                loss_d = gan.hinge_loss_d(real_pred, fake_pred)

                # Add gradient penalty
                gp = gan.gradient_penalty(batch, fake_images.detach(), labels)
                loss_d += 10 * gp  # lambda = 10

                # Add orthogonal regularization
                loss_d += 1e-4 * gan.orthogonal_regularization(gan.discriminator)

            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)

        # Generator training
        with autocast():
            optimizer_g.zero_grad()
            fake_images, fake_labels = gan.generate_fake(batch_size)
            fake_pred = gan.discriminate(fake_images, fake_labels)
            loss_g = gan.hinge_loss_g(fake_pred)

            # Add orthogonal regularization
            loss_g += 1e-4 * gan.orthogonal_regularization(gan.generator)

        scaler.scale(loss_g).backward()
        scaler.step(optimizer_g)

        scaler.update()

        total_loss_d += loss_d.item()
        total_loss_g += loss_g.item()
        batch_idx += 1

    return total_loss_d / batch_idx, total_loss_g / batch_idx


def test(
    gan: BigGAN,
    testloader: DataLoader,
    filename: str,
    epoch: int,
):
    gan.eval()
    with torch.no_grad():
        samples = gan.generate_fake(64)[0]
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
    optimizer_d = Adam(
        biggan.discriminator.parameters(), lr=args.lr_d, betas=(0.0, 0.999)
    )
    optimizer_g = Adam(biggan.generator.parameters(), lr=args.lr_g, betas=(0.0, 0.999))
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=args.epochs)
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=args.epochs)
    scaler = GradScaler()

    loss_train_arr_d = []
    loss_test_arr_d = []
    loss_train_arr_g = []
    loss_test_arr_g = []

    for epoch in range(1, args.epochs + 1):
        loss_train_d, loss_train_g = train(
            biggan,
            trainloader,
            optimizer_d=optimizer_d,
            optimizer_g=optimizer_g,
            scaler=scaler,
        )
        loss_train_arr_d.append(loss_train_d)
        loss_train_arr_g.append(loss_train_g)
        loss_test_d, loss_test_g = test(
            biggan,
            testloader,
            filename,
            epoch,
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
        "--dataset", help="dataset to be modeled.", type=str, default="cifar"
    )
    parser.add_argument(
        "--batch_size", help="number of images in a mini-batch.", type=int, default=128
    )
    parser.add_argument(
        "--epochs", help="maximum number of iterations.", type=int, default=30
    )
    parser.add_argument(
        "--sample_size", help="number of images to generate.", type=int, default=64
    )
    parser.add_argument("--latent-dim", help="latent dimension", type=int, default=100)
    parser.add_argument(
        "--lr-d", help="discriminator learning rate.", type=float, default=4e-4
    )
    parser.add_argument(
        "--lr-g", help="generator learning rate.", type=float, default=1e-4
    )

    args = parser.parse_args()

    # create samples and models folder iff they don't exist:
    if not os.path.exists("samples"):
        os.makedirs("samples")
    if not os.path.exists("models"):
        os.makedirs("models")
    main(args)
