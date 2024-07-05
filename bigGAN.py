import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


image_size = 28
nc = 3  # Number of channels in the training images. For color images this is 3
feature_num = 128  # Size of feature maps in generator/discriminator


# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, img_channels):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.conv_blocks[0:3](out)
        out = self.conv_blocks[3:7](out)
        img = self.conv_blocks[7:](out)
        return img


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        return out


# Initialize the model
class BigGAN:
    def __init__(self, latent_dim, img_size, img_channels, device):
        self.generator = Generator(latent_dim, img_size, img_channels).to(device)
        self.discriminator = Discriminator(img_size, img_channels).to(device)
        self.latent_dim = latent_dim
        self.device = device

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

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
