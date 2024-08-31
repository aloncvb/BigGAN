import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights


def get_inception_model(half=True) -> nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    return model.eval()
