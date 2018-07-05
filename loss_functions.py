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

class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, realIm, fakeIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss



