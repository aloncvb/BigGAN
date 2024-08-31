import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter as P
from torch.utils.checkpoint import checkpoint
from torchvision.models import inception_v3, Inception_V3_Weights


class WrapInception(nn.Module):
    def __init__(self, net):
        super(WrapInception, self).__init__()
        self.net = net
        self.mean = P(
            torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1), requires_grad=False
        )
        self.std = P(
            torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1), requires_grad=False
        )

    def forward(self, x):
        def inception_wrap(x):
            with torch.cuda.amp.autocast():
                x = (x + 1.0) / 2.0
                x = (x - self.mean) / self.std
                if x.shape[2] != 299 or x.shape[3] != 299:
                    x = F.interpolate(
                        x, size=(299, 299), mode="bilinear", align_corners=True
                    )
                x = self.net.Conv2d_1a_3x3(x)
                x = self.net.Conv2d_2a_3x3(x)
                x = self.net.Conv2d_2b_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                x = self.net.Conv2d_3b_1x1(x)
                x = self.net.Conv2d_4a_3x3(x)
                x = F.max_pool2d(x, kernel_size=3, stride=2)
                x = self.net.Mixed_5b(x)
                x = self.net.Mixed_5c(x)

                x = self.net.Mixed_5d(x)
                x = self.net.Mixed_6a(x)
                x = self.net.Mixed_6b(x)
                x = self.net.Mixed_6c(x)
                x = self.net.Mixed_6d(x)
                x = self.net.Mixed_6e(x)
                x = self.net.Mixed_7a(x)
                x = self.net.Mixed_7b(x)
                x = self.net.Mixed_7c(x)
                return x

        x = checkpoint(inception_wrap, x, use_reentrant=False)

        pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
        return pool, logits


def get_inception_model(half=True) -> nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model = WrapInception(model)
    if half:
        return model.eval().half()
    return model.eval()
