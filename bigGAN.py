import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


image_size = 28
nc = 3  # Number of channels in the training images. For color images this is 3
feature_num = 128  # Size of feature maps in generator/discriminator


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, img_channels):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 64, 3, stride=1, padding=1)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, img_channels, 3, stride=1, padding=1)),
            nn.Tanh(),
        )

        self.attn1 = SelfAttention(128)
        self.attn2 = SelfAttention(64)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        out = self.conv_blocks[0:3](out)
        out = self.attn1(out)
        out = self.conv_blocks[3:7](out)
        out = self.attn2(out)
        img = self.conv_blocks[7:](out)
        return img


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_size, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 64, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(128),
            spectral_norm(nn.Conv2d(128, 128, 3, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(128),
        )
        ds_size = img_size // 8
        self.adv_layer = spectral_norm(nn.Linear(128 * ds_size**2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# Initialize the model
class BigGAN:
    def __init__(self, latent_dim, img_size, img_channels, device):
        self.generator = Generator(latent_dim, img_size, img_channels).to(device)
        self.discriminator = Discriminator(img_size, img_channels).to(device)
        self.latent_dim = latent_dim
        self.loss = nn.BCEWithLogitsLoss()
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

    def label(self, x):
        return self.discriminator.forward(x).squeeze()

    def label_real(self, images):
        return self.label(images)

    def label_fake(self, batch_size):
        fake = self.generate_fake(batch_size)
        label = self.label(fake)
        return label

    def calculate_dicriminator_loss(self, real, fake):
        soft_real = torch.full(real.size(), 1.0, device=self.device)
        soft_fake = torch.full(fake.size(), 0.0, device=self.device)
        check = self.loss(real, soft_real) + self.loss(fake, soft_fake)
        return check

    def calculate_generator_loss(self, dis_label):
        soft_real = torch.full(dis_label.size(), 1.0, device=self.device)
        return self.loss(dis_label, soft_real)
