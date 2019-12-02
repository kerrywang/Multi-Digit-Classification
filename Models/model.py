import glob
import os

import torch
import torch.jit
import torch.nn as nn
from torchvision import models

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.checkpoint_template = "BASE_MODEL_{}"

    def store(self, path_to_dir, step):
        checkpoint_file = os.path.join(path_to_dir, self.checkpoint_template.format(step))
        torch.save(self.state_dict(), checkpoint_file)
        return checkpoint_file

    def restore(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))