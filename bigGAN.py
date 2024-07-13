import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_norm(module):
    return nn.utils.spectral_norm(module)


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = spectral_norm(nn.Embedding(num_classes, num_features * 2))

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1
        )
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_classes, upsample=False):
        super().__init__()
        self.cbn1 = ConditionalBatchNorm2d(in_channel, num_classes)
        self.upsample = upsample
        self.conv1 = spectral_norm(nn.Conv2d(in_channel, out_channel, 3, padding=1))
        self.cbn2 = ConditionalBatchNorm2d(out_channel, num_classes)
        self.conv2 = spectral_norm(nn.Conv2d(out_channel, out_channel, 3, padding=1))
        self.bypass = spectral_norm(nn.Conv2d(in_channel, out_channel, 1, padding=0))

    def forward(self, x, y):
        residual = x
        out = self.cbn1(x, y)
        out = F.relu(out)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
            residual = F.interpolate(residual, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = self.cbn2(out, y)
        out = F.relu(out)
        out = self.conv2(out)
        if self.upsample:
            residual = self.bypass(residual)
        return out + residual


class Generator(nn.Module):
    def __init__(self, latent_dim, shared_dim, num_classes, img_size, channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.channels = channels

        self.shared_embed = spectral_norm(nn.Embedding(num_classes, shared_dim))
        self.linear = spectral_norm(nn.Linear(latent_dim + shared_dim, 4 * 4 * 256))
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(256, 256, num_classes, upsample=True),
                ResBlock(256, 256, num_classes, upsample=True),
                ResBlock(256, 128, num_classes, upsample=True),
                ResBlock(128, 64, num_classes, upsample=True),
            ]
        )
        self.final_bn = ConditionalBatchNorm2d(64, num_classes)
        self.final_conv = spectral_norm(nn.Conv2d(64, channels, 3, padding=1))

    def forward(self, z, y):
        shared = self.shared_embed(y)
        out = torch.cat((z, shared), dim=1)
        out = self.linear(out).view(-1, 256, 4, 4)
        for block in self.res_blocks:
            out = block(out, y)
        out = F.relu(self.final_bn(out, y))
        out = self.final_conv(out)
        return torch.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size, channels):
        super().__init__()

        def discriminator_block(in_filters, out_filters):
            return [
                spectral_norm(
                    nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.model = nn.Sequential(
            *discriminator_block(channels, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512, 1024)
        )

        self.linear = spectral_norm(nn.Linear(1024 * (img_size // 32) ** 2, 1))
        self.embed = spectral_norm(
            nn.Embedding(num_classes, 1024 * (img_size // 32) ** 2)
        )

    def forward(self, img, labels):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        output = self.linear(out)
        embed = self.embed(labels)
        prod = (out * embed).sum(1).unsqueeze(1)
        return output + prod


class BigGAN:
    def __init__(
        self, latent_dim, shared_dim, num_classes, img_size, img_channels, device
    ):
        self.generator = Generator(
            latent_dim, shared_dim, num_classes, img_size, img_channels
        ).to(device)
        self.discriminator = Discriminator(num_classes, img_size, img_channels).to(
            device
        )
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device
        self.truncation = 1.0  # Truncation trick parameter

    def generate_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def generate_fake(self, batch_size, labels=None):
        z = self.generate_latent(batch_size)
        if labels is None:
            labels = torch.randint(
                0, self.num_classes, (batch_size,), device=self.device
            )
        z = self.truncate_latent(z)  # Apply truncation trick
        return self.generator.forward(z, labels), labels

    def discriminate(self, x, labels):
        return self.discriminator.forward(x, labels)

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def truncate_latent(self, z):
        return self.truncation * z

    def soft_labels(self, tensor, smoothing=0.1):
        return tensor * (1 - smoothing) + smoothing * 0.5

    def calculate_discriminator_loss(
        self, real_images, real_labels, fake_images, fake_labels
    ):
        real_output = self.discriminate(real_images, real_labels)
        fake_output = self.discriminate(fake_images, fake_labels)

        # Soft labels for real and fake
        real_labels_smooth = self.soft_labels(torch.ones_like(real_output))
        fake_labels_smooth = self.soft_labels(torch.zeros_like(fake_output))

        real_loss = F.binary_cross_entropy_with_logits(real_output, real_labels_smooth)
        fake_loss = F.binary_cross_entropy_with_logits(fake_output, fake_labels_smooth)

        return real_loss + fake_loss

    def calculate_generator_loss(self, fake_images, fake_labels):
        fake_output = self.discriminate(fake_images, fake_labels)

        # Use soft labels for generator as well
        real_labels_smooth = self.soft_labels(torch.ones_like(fake_output))

        return F.binary_cross_entropy_with_logits(fake_output, real_labels_smooth)

    def orthogonal_regularization(self, model):
        reg = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                w = param.view(param.size(0), -1)
                reg += torch.sum(
                    (torch.mm(w, w.t()) - torch.eye(w.size(0), device=w.device)) ** 2
                )
        return reg

    def gradient_penalty(self, real_images, fake_images, labels):
        alpha = torch.rand(real_images.size(0), 1, 1, 1, device=self.device)
        interpolated = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(
            True
        )
        d_interpolated = self.discriminate(interpolated, labels)
        grad = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad = grad.view(grad.size(0), -1)
        penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()
        return penalty
