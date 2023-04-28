import torch
from torch import nn
from torchvision.models import resnet, ResNet, resnet50, ResNet50_Weights


class ResnetBlockLM(nn.Module):

    class ResnetBlock(nn.Module):
        def __init__(self, block, layers, weights):
            super().__init__()
            res0 = ResNet(block, layers)
            if weights is not None:
                res0.load_state_dict(weights.get_state_dict(progress=True))
            self._layers = nn.ModuleList([res0 for i in range(1)])

        def forward(self, x):
            for m in self._layers:
                x = m(x)
            return x

    def __init__(self, block, layers, weights):
        super().__init__()
        self._model = self.ResnetBlock(block, layers, weights)

    def forward(self, x):
        return self._model(x)
