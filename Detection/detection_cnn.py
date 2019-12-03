import torch
import torch.nn as nn

class DigitDetectionClassifier(torch.Module):
    def __init__(self):
        super(DigitDetectionClassifier, self).__init__()

        self.feature = {

        }

        self.classifier = nn.Sequencial(
            nn.Linear(512, )
        )
