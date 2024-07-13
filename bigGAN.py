import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_norm(module):
    return nn.utils.spectral_norm(module)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample=False):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channel, out_channel, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channel, out_channel, 3, padding=1))
        self.bypass_conv = spectral_norm(
            nn.Conv2d(in_channel, out_channel, 1, padding=0)
        )
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.upsample = upsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(x))
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
            residual = F.interpolate(residual, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.upsample:
            residual = self.bypass_conv(residual)
        return out + residual


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size, channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels

        self.init_size = img_size // 4
        self.linear = spectral_norm(nn.Linear(latent_dim, 4 * 4 * 256))
        self.embed = nn.Embedding(num_classes, latent_dim)

        self.res_blocks = nn.ModuleList(
            [
                ResBlock(256, 256, upsample=True),
                ResBlock(256, 128, upsample=True),
            ]
        )
        if img_size == 32:  # For CIFAR
            self.res_blocks.append(ResBlock(128, 64))

        self.final_conv = spectral_norm(
            nn.Conv2d(64 if img_size == 32 else 128, channels, 3, padding=1)
        )

    def forward(self, z, y):
        embedded_y = self.embed(y)
        z = torch.mul(z, embedded_y)
        out = self.linear(z).view(-1, 256, 4, 4)
        for block in self.res_blocks:
            out = block(out)
        out = F.relu(out)
        out = self.final_conv(out)
        return torch.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size, channels):
        super().__init__()

        def discriminator_block(in_filters, out_filters):
            return [
                spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]

        self.model = nn.Sequential(
            *discriminator_block(channels, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        ds_size = img_size // 2**4
        self.adv_layer = spectral_norm(nn.Linear(512 * ds_size**2, 1))
        self.embed = nn.Embedding(num_classes, 512 * ds_size**2)

    def forward(self, img, labels):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        projection = torch.sum(out * self.embed(labels), dim=1, keepdim=True)
        return validity + projection


# Initialize the model
class BigGAN:
    def __init__(self, latent_dim, num_classes, img_size, img_channels, device):
        self.generator = Generator(latent_dim, num_classes, img_size, img_channels).to(
            device
        )
        self.discriminator = Discriminator(img_size, num_classes, img_channels).to(
            device
        )
        self.latent_dim = latent_dim
        self.loss = nn.BCEWithLogitsLoss()
        self.device = device

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def label_smoothing(self, tensor, amount=0.1):
        return tensor * (1 - amount) + amount * 0.5

    def generate_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def generate_fake(self, batch_size, noise=None):
        if noise is not None:
            return self.generator.forward(noise)
        return self.generator.forward(self.generate_latent(batch_size))

    def discriminator_loss(self, real_preds, fake_preds):
        real_loss = torch.mean(F.relu(1.0 - real_preds))
        fake_loss = torch.mean(F.relu(1.0 + fake_preds))
        return real_loss + fake_loss

    def generator_loss(self, fake_preds):
        return -torch.mean(fake_preds)

    def label(self, x):
        return self.discriminator.forward(x)

    def label_real(self, images):
        return self.label(images)

    def label_fake(self, batch_size):
        fake = self.generate_fake(batch_size)
        label = self.label(fake)
        return label

    def calculate_discriminator_loss(self, real, fake):
        # Apply label smoothing
        soft_real = self.label_smoothing(torch.ones_like(real))
        soft_fake = self.label_smoothing(torch.zeros_like(fake))
        check = self.loss(real, soft_real) + self.loss(fake, soft_fake)
        return check

    def calculate_generator_loss(self, dis_label):
        soft_real = torch.full(dis_label.size(), 1.0, device=self.device)
        return self.loss(dis_label, soft_real)
