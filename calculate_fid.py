import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import numpy as np
from bigGANSimple import BigGAN

from inception import InceptionV3


def calculate_fid(mu1, sigma1, mu2, sigma2):
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_statistics(images, inception, device):
    print(images.size())
    inception.eval()
    with torch.no_grad():
        activations = inception(images.to(device)).cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def get_dataloader(dataset_name, batch_size=128):
    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        dataset = datasets.MNIST(
            root="./data/MNIST", train=True, download=True, transform=transform
        )
    elif dataset_name == "cifar":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        dataset = datasets.CIFAR10(
            root="./data/Cifar10", train=True, download=True, transform=transform
        )
    else:
        raise ValueError("Dataset not supported: choose 'mnist' or 'cifar'")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader


def load_generator(dataset_name, device, latent_dim, num_classes):
    # Modify this depending on how your generator is defined
    biggan = BigGAN(
        latent_dim=latent_dim,
        num_classes=num_classes,
        img_channels=1 if dataset_name == "mnist" else 3,
        device=device,
    )
    # if generator path has checkpoint inside , load cifar else load mnist:
    if dataset_name == "cifar":
        checkpoint = torch.load("checkpoint_cifar.pt", map_location=device)
        biggan.generator.load_state_dict(checkpoint["generator_state_dict"])
        print("Loaded generator for cifar")
    else:
        checkpoint = torch.load("checkpoint.pt", map_location=device)
        biggan.generator.load_state_dict(checkpoint["generator_state_dict"])
        print("Loaded generator for mnist")

    biggan.eval()
    return biggan.generator


def generate_images(generator, num_images, latent_dim, num_classes, device):
    z = torch.randn(num_images, latent_dim, device=device)
    labels = torch.randint(0, num_classes, (num_images,), device=device)
    with torch.no_grad():
        generated_images = generator(z, labels).cpu()
    return generated_images


def main(
    dataset_name,
    latent_dim=128,
    num_classes=10,
    num_images=10000,
    batch_size=128,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    generator = load_generator(dataset_name, device, latent_dim, num_classes)

    dataloader = get_dataloader(dataset_name, batch_size)

    inception = InceptionV3(device=device)

    generated_images = generate_images(
        generator, num_images, latent_dim, num_classes, device
    )

    real_images = next(iter(dataloader))[0][:num_images]
    mu_real, sigma_real = compute_statistics(real_images, inception, device)

    mu_gen, sigma_gen = compute_statistics(generated_images, inception, device)

    fid_score = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f"FID score for {dataset_name}: {fid_score}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate FID for a GAN generator.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["mnist", "cifar"],
        required=True,
        help="Dataset name: 'mnist' or 'cifar'.",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="Latent dimension size."
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Number of classes in the dataset."
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10000,
        help="Number of images to generate for FID calculation.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for data loading."
    )

    args = parser.parse_args()
    main(
        args.dataset_name,
        args.latent_dim,
        args.num_classes,
        args.num_images,
        args.batch_size,
    )
