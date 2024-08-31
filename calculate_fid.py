import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths

from bigGANSimple import BigGAN  # Assuming your model is defined here


def save_images(
    dataset_name,
    generator,
    num_images,
    latent_dim,
    num_classes,
    batch_size,
    device,
):
    if not os.path.exists("real"):
        os.makedirs("real")
        os.makedirs("fake")

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

    # Save real images
    for i, (real_images, _) in enumerate(dataloader):
        if i * batch_size >= num_images:
            break
        for j in range(real_images.size(0)):
            save_image(
                real_images[j],
                os.path.join("real", f"real_{i * batch_size + j}.png"),
            )

    # Save fake images
    generator.eval()
    with torch.no_grad():
        for i in range(num_images // batch_size):
            z = torch.randn(batch_size, latent_dim, device=device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_images = generator(z, labels).cpu()
            for j in range(fake_images.size(0)):
                save_image(
                    fake_images[j],
                    os.path.join("fake", f"fake_{i * batch_size + j}.png"),
                )


def main(
    generator_path,
    dataset_name,
    latent_dim=128,
    num_classes=10,
    num_images=10000,
    batch_size=128,
    device="cuda:0",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load the generator
    biggan = BigGAN(
        latent_dim=latent_dim,
        num_classes=num_classes,
        img_channels=1 if dataset_name == "mnist" else 3,
        device=device,
    )
    if dataset_name == "mnist":
        checkpoint = torch.load("checkpoint.pt", map_location=device)
        biggan.generator.load_state_dict(checkpoint["generator_state_dict"])
        print("loaded mnist model")
    else:
        checkpoint = torch.load("checkpoint_cifar.pt", map_location=device)
        biggan.generator.load_state_dict(checkpoint["generator_state_dict"])
        print("loaded model")

    biggan.to(device)

    print("CREATING IMAGES")
    # Save real and fake images
    save_images(
        dataset_name,
        biggan.generator,
        num_images,
        latent_dim,
        num_classes,
        batch_size,
        device,
    )

    paths = [os.path.join("real"), os.path.join("fake")]

    # Calculate FID Score over all dataset
    print("Calculating FID Score")
    fid_value = calculate_fid_given_paths(
        paths, batch_size=256, device=device, dims=2048
    )

    print(f"FID Score: {fid_value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and save real and fake images for FID calculation."
    )
    parser.add_argument(
        "--generator_path",
        type=str,
        default="checkpoint.pt",
        help="Path to the generator model.",
    )
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
        "--num_images", type=int, default=10000, help="Number of images to generate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for data loading."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for generating images.",
    )

    args = parser.parse_args()
    main(
        args.generator_path,
        args.dataset_name,
        args.latent_dim,
        args.num_classes,
        args.num_images,
        args.batch_size,
        args.device,
    )
