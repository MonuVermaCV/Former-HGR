import torch
from torch import nn
from models.HGR_Former import spatial_transformer


class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.HGR_former = spatial_transformer()
        self.drop =nn.Dropout(p=0.5)

        
        self.fc1 = nn.Linear(18816, 512)
        self.fc11 = nn.Linear(512, 10)
        

    def forward(self, x):

        x = self.HGR_former(x)

        x_HGR =x
        y1 = self.fc1(x)
        y =self.drop(y1)
        y = self.fc11(y)
        
        return y, x_HGR


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    model(img)
