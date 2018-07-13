import torch
import torch.nn as nn 

class Generator(nn.Module):
    
    def __init__(self, input_dim, output_dim, input_size):
        super(self, Generator).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        
        self.fc1 = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(input_dim, 64, kernel_size=7),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
        
        self.fc2 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )

        self.fc3 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
        
        self.resblock = ResBlock(inplac=256, planes=256)

        self.fc4 = 
    
