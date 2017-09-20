"""
Attentive CNN
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        #