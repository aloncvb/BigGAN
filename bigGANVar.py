import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# used as a non-local block as described in the paper
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, W, H = x.size()
        query = self.query_conv(x).view(batch_size, -1, W * H).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, W * H)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, W * H)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, W, H)
        return self.gamma * out + x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.upsample = upsample
        self.skip_connection = (
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1))
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(x), 0.2)
        out = self.conv1(out)
        out = F.leaky_relu(self.bn2(out), 0.2)
        out = self.conv2(out)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
            residual = F.interpolate(residual, scale_factor=2, mode="nearest")
        residual = self.skip_connection(residual)
        return out + residual


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 2 * 2 * 448)
        self.bn1 = nn.BatchNorm1d(2 * 2 * 448)
        self.deconv2 = nn.ConvTranspose2d(448, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = F.relu(self.bn1(self.fc1(z)))
        x = x.view(x.size(0), 448, 2, 2)
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        out = torch.sigmoid(self.deconv5(x))
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 1024, kernel_size=4, stride=4)
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(x.size(0), -1)
        out_logit = self.fc(x)
        out = torch.sigmoid(out_logit)
        return out, out_logit, x


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if hasattr(m, "weight"):
            nn.init.orthogonal_(m.weight)  # maybe xavier?
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BigGAN(nn.Module):
    def __init__(self, latent_dim, num_classes, img_channels, device, ch=32):
        super(BigGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.generator = Generator(62).to(device)
        self.discriminator = Discriminator().to(device)
        self.device = device

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    def generate_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def generate_fake(self, batch_size, labels=None):
        z = self.generate_latent(batch_size)
        if labels is None:
            labels = torch.randint(
                0, self.num_classes, (batch_size,), device=self.device
            )
        # z = self.truncate_latent(z)  # Apply truncation trick

        images = self.generator.forward(z, labels)
        return images, labels

    def discriminate(self, x, labels):
        return self.discriminator.forward(x, labels)

    def train(self):
        self.generator.train()
        self.discriminator.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def truncate_latent(self, z, threshold=0.5):
        norm = torch.norm(z, dim=1, keepdim=True)
        return z * (norm < threshold).float()

    def soft_labels(self, tensor, smoothing=0.1):
        return tensor * (1 - smoothing) + smoothing * 0.5

    def calculate_discriminator_loss(
        self, real_images, real_labels, fake_images, fake_labels
    ):
        real_output = self.discriminate(real_images, real_labels)
        fake_output = self.discriminate(fake_images, fake_labels)

        # Soft labels for real and fake
        real_labels_smooth = self.soft_labels(
            torch.ones_like(real_output), smoothing=0.1
        )
        fake_labels_smooth = self.soft_labels(
            torch.zeros_like(fake_output), smoothing=0.1
        )

        real_loss = F.binary_cross_entropy_with_logits(real_output, real_labels_smooth)
        fake_loss = F.binary_cross_entropy_with_logits(fake_output, fake_labels_smooth)

        return real_loss + fake_loss

    def calculate_generator_loss(self, fake_images, fake_labels):
        fake_output = self.discriminate(fake_images, fake_labels)

        # Use soft labels for generator as well
        real_labels_smooth = self.soft_labels(
            torch.ones_like(fake_output), smoothing=0.1
        )

        return F.binary_cross_entropy_with_logits(fake_output, real_labels_smooth)

    def generate_high_quality(self, batch_size, truncation=0.5):
        with torch.no_grad():
            z = self.generate_latent(batch_size)
            labels = torch.randint(
                0, self.num_classes, (batch_size,), device=self.device
            )
            z = z * truncation

            images = self.generator.forward(z, labels)
            return images

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
