import torch
import torch.nn as nn
from split import set_seed
from torch.nn.modules import dropout

class MyCNN(nn.Module):
    def __init__(self,
                input_channels: int = 1,
                hidden_channels: list = [32, 64, 64, 64, 128, 128, 256, 128, 128, 64, 64, 64, 32],
                use_batchnormalization: bool = True,
                num_classes: int = 20,
                kernel_size: list = [5,5,3,3,3,3,3,3,3,3,3,3,3],
                activation_function: torch.nn.Module = torch.nn.ReLU()):
        super().__init__()
        layers = []
        pooling_layers = 0
        in_features=input_channels
        for i, channel in enumerate(hidden_channels):
            layer = nn.Conv2d(in_channels=in_features, 
                              out_channels=channel, 
                              kernel_size=kernel_size[i], 
                              padding=kernel_size[i]//2)
            layers.append(layer)
            layers.append(activation_function)
            
            if i%3==0:
              layers.append(nn.MaxPool2d(2))
              pooling_layers+=1

            if use_batchnormalization:
                layers.append(nn.BatchNorm2d(channel))

            in_features = channel

        self.layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten() 
        self.dropout = nn.Dropout(p=0.5)

        out_size = 100 # it's hardcoded, but since all the pictures are 100*100, I don't think it matters
        for i in range(pooling_layers):
            out_size = (out_size-2)//2 + 1 

        self.output_layer = nn.Linear(in_features=in_features*out_size*out_size, out_features=num_classes)
    
    def forward(self, input_images: torch.Tensor):
        x = self.layers(input_images)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.output_layer(x)

set_seed(333)        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyCNN().to(device=device)