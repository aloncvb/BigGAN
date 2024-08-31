import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class InceptionV3(nn.Module):
    def __init__(self, device="cpu"):
        super(InceptionV3, self).__init__()
        self.inception = models.inception_v3(pretrained=True, transform_input=False)
        # We only need the layers up to the last pooling layer (before the final classifier)
        self.inception = nn.Sequential(*list(self.inception.children())[:-1])
        self.inception.to(device)
        self.device = device

    def forward(self, x):
        # Ensure input size is (N, 3, 299, 299) as expected by InceptionV3
        if x.size(1) != 3 or x.size(2) != 299 or x.size(3) != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        else:
            print(x.size())
        x = self.inception(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        return x
