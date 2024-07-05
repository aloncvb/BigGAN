"""Training procedure for gan.
"""

import os
import argparse
from random import randrange
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam

# from dcgan import DCGAN


def train(
    dcgan: DCGAN,
    trainloader: DataLoader,
    trainloader1000: DataLoader,
    optimizer_d: Adam,
    optimizer_g: Adam,
    epoch: int,
    learning_way: str = "reg",
):
    dcgan.train()  # set to training mode

    total_loss_d = 0
    total_loss_g = 0
    batch_idx = 0
    best_fid = float("inf")
    old_fid = 0
    real_data = []
    update_flag = True
    for batch, _ in trainloader1000:
        real_data.append(batch.to(dcgan.device))
        if len(real_data) == 10:
            break
    for batch, _ in trainloader:
        data = batch.to(dcgan.device)
        optimizer_d.zero_grad()
        batch_size = data.size()[0]
        # discriminator train
        real_label = dcgan.label_real(data)
        fake_label = dcgan.label_fake(batch_size=batch_size)
        loss_d = dcgan.calculate_dicriminator_loss(real_label, fake_label)
        loss_d.backward()

        total_loss_d += loss_d.item()
        optimizer_d.step()

        # generator train
        optimizer_g.zero_grad()
        fake_images = dcgan.generate_fake(batch_size)
        results = dcgan.label(fake_images)
        loss_g = dcgan.calculate_generator_loss(results)
        if learning_way == "reverse" and epoch > 9:
            # random number between 1 to 10
            rand_num = randrange(1, 10)
            real_mu, real_sigma = get_activation_statistics(
                real_data[rand_num], device=dcgan.device
            )
            fake_images_fid = dcgan.generate_fake(1000)  # 1000 for stable score
            fake_mu, fake_sigma = get_activation_statistics(
                fake_images_fid, device=dcgan.device
            )
            new_fid = frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

            if new_fid.item() < best_fid:
                best_fid = new_fid.item()
                best_generator = dcgan.generator.state_dict()
                dcgan.generator.load_state_dict(best_generator)
                optimizer_g = torch.optim.Adam(
                    dcgan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
                )
                update_flag = False
        elif learning_way == "rl":
            real_mu, real_sigma = get_activation_statistics(data, device=dcgan.device)
            fake_images_fid = dcgan.generate_fake(batch_size)
            fake_mu, fake_sigma = get_activation_statistics(
                fake_images_fid, device=dcgan.device
            )
            new_fid = frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
            reward = get_reward_loss(old_fid, new_fid)
            old_fid = new_fid
            if old_fid != 0:
                loss_g = loss_g + reward

        elif learning_way == "fid":
            real_mu, real_sigma = get_activation_statistics(
                data,
                device=dcgan.device,
            )
            fake_images_fid = dcgan.generate_fake(batch_size)
            # use fid for better training
            fake_mu, fake_sigma = get_activation_statistics(
                fake_images_fid,
                device=dcgan.device,
            )
            fid_loss = frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)
            loss_g = 0.7 * loss_g + 0.3 * fid_loss

        if update_flag:
            loss_g.backward()
            total_loss_g += loss_g.item()
            optimizer_g.step()
        else:
            update_flag = True

        batch_idx += 1
    return total_loss_d / batch_idx, total_loss_g / batch_idx


def test(
    dcgan: DCGAN,
    testloader: DataLoader,
    filename: str,
    epoch: int,
    fixed_noise,
    learning_way: str = "reg",
):
    dcgan.eval()  # set to inference mode
    with torch.no_grad():
        samples = dcgan.generate_fake(100, fixed_noise[:100])
        torchvision.utils.save_image(
            torchvision.utils.make_grid(samples),
            "./samples/" + filename + "epoch%d.png" % epoch,
        )

        total_loss_g = 0
        total_loss_d = 0
        batch_idx = 0
        for index, (batch, _) in enumerate(testloader):
            data = batch.to(dcgan.device)
            batch_size = data.size()[0]
            real_label = dcgan.label_real(data)
            fake_label = dcgan.label_fake(batch_size=batch_size)
            loss_d = dcgan.calculate_dicriminator_loss(real_label, fake_label)
            total_loss_d += loss_d.item()

            fake_images = dcgan.generate_fake(
                batch_size,
                noise=fixed_noise[index * batch_size : (index + 1) * batch_size],
            )
            results = dcgan.label(fake_images)
            loss_g = dcgan.calculate_generator_loss(results)
            if batch_idx % 1 == 0:
                if learning_way == "fid":
                    real_mu, real_sigma = get_activation_statistics(
                        data,
                        device=dcgan.device,
                    )
                    fake_images_fid = dcgan.generate_fake(
                        batch_size
                    )  # 1000 for stable score
                    # use fid for better training
                    fake_mu, fake_sigma = get_activation_statistics(
                        fake_images_fid,
                        device=dcgan.device,
                    )
                    fid_loss = frechet_distance(
                        real_mu, real_sigma, fake_mu, fake_sigma
                    )
                    loss_g = 0.7 * loss_g + 0.3 * fid_loss

            total_loss_g += loss_g.item()
            batch_idx += 1
        print(
            "Epoch: {} Test set: Average loss_d: {:.4f}".format(
                epoch, total_loss_d / batch_idx
            )
        )
        print(
            "Epoch: {} Test set: Average loss_g: {:.4f}".format(
                epoch, total_loss_g / batch_idx
            )
        )
    return total_loss_d / batch_idx, total_loss_g / batch_idx


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mnist":
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (64),
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
        trainloader1000 = torch.utils.data.DataLoader(
            trainset, batch_size=1000, shuffle=True, num_workers=2
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
                transforms.Resize(
                    (64),
                    interpolation=transforms.InterpolationMode.BICUBIC,  # size_that_worked = 64
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
        trainloader1000 = torch.utils.data.DataLoader(
            trainset, batch_size=1000, shuffle=True, num_workers=2
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
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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

    dcgan = DCGAN(latent_dim=args.latent_dim, device=device)
    optimizer_d = torch.optim.Adam(
        dcgan.discriminator.parameters(), lr=args.lr * 2, betas=(0.5, 0.999)
    )
    optimizer_g = torch.optim.Adam(
        dcgan.generator.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    loss_train_arr_d = []
    loss_test_arr_d = []
    loss_train_arr_g = []
    loss_test_arr_g = []
    fixed_noise = torch.randn(testset.__len__(), args.latent_dim, 1, 1, device=device)
    for epoch in range(1, args.epochs + 1):
        loss_train_d, loss_train_g = train(
            dcgan,
            trainloader,
            trainloader1000,
            optimizer_d=optimizer_d,
            optimizer_g=optimizer_g,
            epoch=epoch,
            learning_way=args.lw,
        )
        loss_train_arr_d.append(loss_train_d)
        loss_train_arr_g.append(loss_train_g)
        loss_test_d, loss_test_g = test(
            dcgan, testloader, filename, epoch, fixed_noise, args.lw
        )
        loss_test_arr_d.append(loss_test_d)
        loss_test_arr_g.append(loss_test_g)
    # Save the model
    torch.save(dcgan.generator.state_dict(), "generator.pt")
    torch.save(dcgan.discriminator.state_dict(), "discriminator.pt")
    # create a plot of the loss
    plt.plot(loss_train_arr_d, label="train_d")
    plt.plot(loss_test_arr_d, label="test_d")
    plt.plot(loss_train_arr_g, label="train_g")
    plt.plot(loss_test_arr_g, label="test_g")
    plt.xlabel("Epoch")
    plt.ylabel("dcgan loss")
    plt.legend()
    plt.savefig("loss.png")


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

    parser.add_argument("--lw", help="way of learning.", type=str, default="reg")

    args = parser.parse_args()

    # create samples and models folder iff they don't exist:
    if not os.path.exists("samples"):
        os.makedirs("samples")
    if not os.path.exists("models"):
        os.makedirs("models")
    # main(args)
