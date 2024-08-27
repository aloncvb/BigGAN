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


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, c, h, w = x.size()

        # Theta operation
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # (B, C', N)
        theta_x = theta_x.permute(0, 2, 1)  # (B, N, C')

        # Phi operation
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (B, C', N)

        # Calculate attention map
        f = torch.matmul(theta_x, phi_x)  # (B, N, N)
        f_div_C = F.softmax(f, dim=-1)  # (B, N, N)

        # G operation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (B, C', N)

        # Apply attention map to g_x
        y = torch.matmul(f_div_C, g_x.permute(0, 2, 1))  # (B, N, C')
        y = y.permute(0, 2, 1).contiguous()  # (B, C', N)
        y = y.view(batch_size, self.inter_channels, h, w)  # (B, C', H, W)

        W_y = self.W(y)
        z = W_y + x

        return z


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
    def __init__(self, latent_dim, num_classes, ch=64, img_channels=3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.ch = ch

        self.embed = spectral_norm(nn.Embedding(num_classes, 128))
        self.linear = spectral_norm(nn.Linear(latent_dim + 128, 4 * 4 * 16 * ch))
        self.res1 = ResBlock(16 * ch, 16 * ch, upsample=True)
        self.res2 = ResBlock(16 * ch, 8 * ch, upsample=True)
        self.res3 = ResBlock(8 * ch, 4 * ch, upsample=True)
        self.res4 = ResBlock(4 * ch, 2 * ch, upsample=False)
        self.attention = SelfAttention(2 * ch)
        self.non_local = NonLocalBlock(2 * ch)
        self.res5 = ResBlock(2 * ch, ch, upsample=False)
        self.bn = nn.BatchNorm2d(ch)
        self.conv_out = spectral_norm(nn.Conv2d(ch, img_channels, 3, padding=1))
        self.dropout = nn.Dropout(0.4)

    def forward(self, z, y):
        y_embed = self.embed(y)
        z = torch.cat([z, y_embed], dim=1)
        x = self.linear(z).view(-1, 16 * self.ch, 4, 4)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        # x = self.attention(x)
        x = self.non_local(x)
        x = self.res5(x)
        x = F.leaky_relu(self.bn(x), 0.2)

        x = self.dropout(x)
        x = torch.tanh(self.conv_out(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, num_classes, ch=32, img_channels=3):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        self.res1 = ResBlock(img_channels, ch, upsample=False)
        self.attention = SelfAttention(ch)
        self.non_local = NonLocalBlock(ch)
        self.res2 = ResBlock(ch, 2 * ch, upsample=False)
        self.res3 = ResBlock(2 * ch, 4 * ch, upsample=False)
        self.res4 = ResBlock(4 * ch, 8 * ch, upsample=False)
        self.res5 = ResBlock(8 * ch, 16 * ch, upsample=False)
        self.res6 = ResBlock(16 * ch, 16 * ch, upsample=False)
        self.linear = spectral_norm(nn.Linear(16 * ch, 1))
        self.embed = spectral_norm(nn.Embedding(num_classes, 16 * ch))
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, y):
        x = self.res1(x)
        # x = self.attention(x)
        x = self.non_local(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = F.leaky_relu(x, 0.2)
        # global sum pooling: sum over all spatial dimensions
        x = torch.sum(x, dim=[2, 3])

        x = self.dropout(x)
        out = self.linear(x)
        embed = self.embed(y)
        prod = torch.sum(x * embed, dim=1, keepdim=True)
        return out + prod


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
        self.generator = Generator(
            latent_dim, num_classes, ch, img_channels=img_channels
        ).to(device)
        self.discriminator = Discriminator(
            num_classes, ch, img_channels=img_channels
        ).to(device)
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
