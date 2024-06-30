import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self,
                input_channels: int = 1,
                hidden_channels: list =[32,64,128],
                use_batchnormalization: bool = True,
                num_classes: int = 20,
                kernel_size: list = [3,5,3],
                activation_function: torch.nn.Module = torch.nn.ReLU()):
        super().__init__()
        layers = []
        in_features=input_channels
        for i, channel in enumerate(hidden_channels):
            layer = nn.Conv2d(in_channels=in_features, 
                              out_channels=channel, 
                              kernel_size=kernel_size[i], 
                              padding=kernel_size[i]//2)
            layers.append(layer)
            layers.append(activation_function)

            if use_batchnormalization:
                layers.append(nn.BatchNorm2d(channel))

            in_features = channel

        self.layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten() 
        self.output_layer = nn.Linear(in_features=in_features*100*100, out_features=num_classes)
    
    def forward(self, input_images: torch.Tensor):
        x = self.layers(input_images)
        x = self.flatten(x)
        return self.output_layer(x)

torch.manual_seed(333)        
model = MyCNN()