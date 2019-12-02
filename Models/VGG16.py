from Models.model import BaseModel
import torch.nn as nn
from torchvision import models
import torch

class VGG16(BaseModel):

    def __init__(self, load_weights=False):
        super(VGG16, self).__init__()
        self.original_vgg = models.vgg16(pretrained=load_weights)
        self.feature = self.original_vgg.features
        if load_weights:
            for param in self.feature.parameters():
                param.require_grad = False

        self.classifiers = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512)
        )


        self._digit_length = nn.Sequential(nn.Linear(512, 7))
        self._digit1 = nn.Sequential(nn.Linear(512, 11))
        self._digit2 = nn.Sequential(nn.Linear(512, 11))
        self._digit3 = nn.Sequential(nn.Linear(512, 11))
        self._digit4 = nn.Sequential(nn.Linear(512, 11))
        self._digit5 = nn.Sequential(nn.Linear(512, 11))

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifiers(x)

        length_logits = self._digit_length(x)
        digit1_logits = self._digit1(x)
        digit2_logits = self._digit2(x)
        digit3_logits = self._digit3(x)
        digit4_logits = self._digit4(x)
        digit5_logits = self._digit5(x)

        return length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits