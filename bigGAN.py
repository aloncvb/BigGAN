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


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, img_channels):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.init_size = img_size // 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(128),
                    nn.Upsample(scale_factor=2),
                    spectral_norm(nn.Conv2d(128, 128, 3, stride=1, padding=1)),
                    nn.BatchNorm2d(128, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    spectral_norm(nn.Conv2d(128, 64, 3, stride=1, padding=1)),
                    nn.BatchNorm2d(64, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    spectral_norm(nn.Conv2d(64, img_channels, 3, stride=1, padding=1)),
                    nn.Tanh(),
                ),
            ]
        )

        self.attn1 = SelfAttention(128)
        self.attn2 = SelfAttention(64)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)

        for i, block in enumerate(self.conv_blocks):
            out = block(out)
            if i == 0:
                out = self.attn1(out)
            elif i == 1:
                out = self.attn2(out)

            if (
                out.size(2) == self.img_size
            ):  # If we've reached the target size, stop upsampling
                break

        return out


class Discriminator(nn.Module):
    def __init__(self, img_size, img_channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_channels, 64, bn=False),
            *discriminator_block(64, 128),
            SelfAttention(128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            SelfAttention(512),
        )

        # The height and width of downsampled image
        ds_size = img_size // 16
        self.adv_layer = spectral_norm(nn.Linear(512 * ds_size**2, 1))

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
