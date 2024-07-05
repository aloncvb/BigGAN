"""DCGAN model
"""

import torch
import torch.nn as nn


image_size = 28
nc = 3  # Number of channels in the training images. For color images this is 3
feature_num = 128  # Size of feature maps in generator/discriminator


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, feature_num * 8, 4, 1, 0),
            nn.BatchNorm2d(feature_num * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_num * 8, feature_num * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_num * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_num * 4, feature_num * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_num * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_num * 2, feature_num, 4, 2, 1),
            nn.BatchNorm2d(feature_num),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_num, nc, 4, 2, 1),
            nn.Tanh(),
            # Output size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is (nc) x 128
            nn.Conv2d(nc, feature_num, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num, feature_num * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_num * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num * 2, feature_num * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_num * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num * 4, feature_num * 8, 4, 2, 1),
            nn.BatchNorm2d(feature_num * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num * 8, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(input)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN:
    def __init__(self, latent_dim, device):
        super(DCGAN, self).__init__()
        self.generator = Generator(latent_dim)
        self.generator.to(device)
        self.discriminator = Discriminator()
        self.discriminator.to(device)
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.device = device
        self.latent_dim = latent_dim
        self.loss = nn.BCELoss()

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def label(self, x):
        return self.discriminator(x).squeeze()

    def generate_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)

    def generate_fake(self, batch_size, noise=None):
        if noise is not None:
            return self.generator(noise).to(self.device)
        return self.generator(self.generate_latent(batch_size))

    def label_real(self, images):
        return self.label(images)

    def label_fake(self, batch_size):
        fake = self.generate_fake(batch_size)
        label = self.label(fake)
        return label

    def calculate_dicriminator_loss(self, real, fake):
        soft_real = torch.full(real.size(), 0.9, device=self.device)
        soft_fake = torch.full(fake.size(), 0.1, device=self.device)
        check = self.loss(real, soft_real) + self.loss(fake, soft_fake)
        return check

    def calculate_generator_loss(self, dis_label):
        soft_real = torch.full(dis_label.size(), 0.9, device=self.device)
        return self.loss(dis_label, soft_real)
