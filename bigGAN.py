import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_norm(module):
    return nn.utils.spectral_norm(module)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv1d(in_channels, in_channels // 8, 1))
        self.key = spectral_norm(nn.Conv1d(in_channels, in_channels // 8, 1))
        self.value = spectral_norm(nn.Conv1d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, W, H = x.size()
        loc = x.view(batch_size, C, -1)

        query = self.query(loc).view(batch_size, -1, W * H).permute(0, 2, 1)
        key = self.key(loc).view(batch_size, -1, W * H)
        value = self.value(loc).view(batch_size, -1, W * H)

        attn = torch.bmm(query, key)
        attn = F.softmax(attn, dim=2)

        out = torch.bmm(value, attn.permute(0, 2, 1))
        out = out.view(batch_size, C, W, H)

        return self.gamma * out + x


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
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, y):
        residual = x
        out = self.activation(self.cbn1(x, y))
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
            residual = F.interpolate(residual, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = self.activation(self.cbn2(out, y))
        out = self.conv2(out)
        if self.upsample:
            residual = self.bypass(residual)
        return out + residual


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size, channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.linear = spectral_norm(nn.Linear(latent_dim, 4 * 4 * 512))
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(512, 256, num_classes, upsample=True),
                ResBlock(256, 128, num_classes, upsample=True),
                ResBlock(128, 64, num_classes, upsample=True),
            ]
        )
        self.attention = SelfAttention(64)
        self.final_bn = ConditionalBatchNorm2d(64, num_classes)
        self.final_conv = spectral_norm(nn.Conv2d(64, channels, 3, padding=1))

    def forward(self, z, y):
        out = self.linear(z).view(-1, 512, 4, 4)
        for block in self.res_blocks:
            out = block(out, y)
        out = self.attention(out)
        out = F.leaky_relu(self.final_bn(out, y), 0.2)
        out = self.final_conv(out)
        return torch.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size, channels):
        super().__init__()

        def discriminator_block(in_filters, out_filters, stride=2):
            return [
                spectral_norm(
                    nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        self.model = nn.Sequential(
            *discriminator_block(channels, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            SelfAttention(256),
            *discriminator_block(256, 512, stride=1),
        )

        self.output_size = 4
        self.output_dim = 512 * self.output_size * self.output_size

        self.linear = spectral_norm(nn.Linear(self.output_dim, 1))
        self.embed = spectral_norm(nn.Embedding(num_classes, self.output_dim))

    def forward(self, img, labels):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        output = self.linear(out)
        embed = self.embed(labels)
        prod = (out * embed).sum(1).unsqueeze(1)
        return output + prod


class BigGAN:
    def __init__(self, latent_dim, num_classes, img_size, img_channels, device):
        self.generator = Generator(latent_dim, num_classes, img_size, img_channels).to(
            device
        )
        self.discriminator = Discriminator(num_classes, img_size, img_channels).to(
            device
        )
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device

    def generate_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def generate_fake(self, batch_size, labels=None):
        z = self.generate_latent(batch_size)
        if labels is None:
            labels = torch.randint(
                0, self.num_classes, (batch_size,), device=self.device
            )
        # z = self.truncate_latent(z)  # Apply truncation trick
        return self.generator.forward(z, labels), labels

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
            torch.ones_like(real_output), smoothing=0.0
        )
        fake_labels_smooth = self.soft_labels(
            torch.zeros_like(fake_output), smoothing=0.0
        )

        real_loss = F.binary_cross_entropy_with_logits(real_output, real_labels_smooth)
        fake_loss = F.binary_cross_entropy_with_logits(fake_output, fake_labels_smooth)

        return real_loss + fake_loss

    def calculate_generator_loss(self, fake_images, fake_labels):
        fake_output = self.discriminate(fake_images, fake_labels)

        # Use soft labels for generator as well
        real_labels_smooth = self.soft_labels(
            torch.ones_like(fake_output), smoothing=0.0
        )

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
