import torch
import torch.nn as nn 
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
from torch.autograd import Variable

class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss
    
    def get_loss(self, realIm, fakeIm):
        return self.criterion(realIm, fakeIm)
