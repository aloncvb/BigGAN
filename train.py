import os
import argparse
from random import randrange
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from bigGAN import BigGAN
import torch.nn.functional as F


def orthogonal_regularization(model):
    loss = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            mat = param.view(param.size(0), -1)
            sym = torch.mm(mat, mat.t())
            sym -= torch.eye(mat.size(0)).to(mat.device)
            loss += sym.abs().sum()
    return loss


def feature_matching_loss(real_features, fake_features):
    loss = 0
    for real_feat, fake_feat in zip(real_features, fake_features):
        loss += F.mse_loss(fake_feat.mean(0), real_feat.mean(0))
    return loss


def train(
    gan: BigGAN,
    trainloader: DataLoader,
    optimizer_d: Adam,
    optimizer_g: Adam,
    scaler: GradScaler,
    max_grad_norm=1.0,
    noise_factor=0.05,
):
    gan.train()
    total_loss_d = 0
    total_loss_g = 0
    batch_idx = 0

    for batch, labels in trainloader:
        batch = batch.to(gan.device)
        batch_size = batch.size()[0]
        labels = labels.to(gan.device)

        # Add noise to real images
        noise = torch.randn_like(batch) * noise_factor
        noisy_batch = batch + noise

        # Discriminator training
        with autocast():
            optimizer_d.zero_grad()
            real_pred, real_features = gan.discriminator(
                noisy_batch, labels, get_features=True
            )
            fake_images, fake_labels = gan.generate_fake(batch_size)

            # Add noise to fake images
            noise = torch.randn_like(fake_images) * noise_factor
            noisy_fake_images = fake_images + noise

            fake_pred, _ = gan.discriminator(
                noisy_fake_images.detach(), fake_labels, get_features=True
            )

            loss_d = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            ) + F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )

            # Gradient penalty
            gp = gan.gradient_penalty(noisy_batch, noisy_fake_images.detach(), labels)
            loss_d += 0.1 * gp

        scaler.scale(loss_d).backward()
        scaler.unscale_(optimizer_d)
        torch.nn.utils.clip_grad_norm_(gan.discriminator.parameters(), max_grad_norm)
        scaler.step(optimizer_d)
        scaler.update()

        # Generator training
        with autocast():
            optimizer_g.zero_grad()
            fake_images, fake_labels = gan.generate_fake(batch_size)
            fake_pred, fake_features = gan.discriminator(
                fake_images, fake_labels, get_features=True
            )

            loss_g = F.binary_cross_entropy_with_logits(
                fake_pred, torch.ones_like(fake_pred)
            )

            # Feature matching loss
            fm_loss = feature_matching_loss(real_features, fake_features)
            loss_g += fm_loss * 0.1  # Adjust this weight as needed

            # Orthogonal regularization
            ortho_reg = orthogonal_regularization(gan.generator)
            loss_g += ortho_reg * 1e-4  # Adjust this weight as needed

        scaler.scale(loss_g).backward()
        scaler.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(gan.generator.parameters(), max_grad_norm)
        scaler.step(optimizer_g)
        scaler.update()

        total_loss_d += loss_d.item()
        total_loss_g += loss_g.item()
        batch_idx += 1

        if batch_idx % 100 == 0:
            print(
                f"Batch {batch_idx}: D loss: {loss_d.item():.4f}, G loss: {loss_g.item():.4f}"
            )

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
                RandomHorizontalFlip(),
                RandomCrop(32, padding=4),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
    parser.add_argument("--latent-dim", help="latent dimension", type=int, default=128)
    parser.add_argument(
        "--lr-d", help="discriminator learning rate.", type=float, default=3e-4
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
