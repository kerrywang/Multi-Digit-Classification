from Models.model import BaseModel
import torch.nn as nn
from torchvision import models
import torch

class VGG16Detection(BaseModel):

    def __init__(self, load_weights=False):
        super(VGG16Detection, self).__init__()
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
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 2),
        )


    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifiers(x)
        return x