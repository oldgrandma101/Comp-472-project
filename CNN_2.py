#this project needs 3 different variants of CNNs to experiment with, this is the second one
#This variant has 6 convolutional layers with 1 linear layer to classify into the 4 classes
#Can also try different activation functions instead of leakyRELU or different pooling than MaxPool2D
#can also experiment with the number of convolutional layers
# can also try different kernel sizes
import torch.nn as nn

class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv_layer = nn.Sequential(    #defines convolutional layers in CNN

            # in_channels = 1 because our images are grayscale
            # layer 1 has 32 kernels
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            # layer 2 has 64 kernels
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # layer 3 has 128 kernels
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            # layer 4 has 256 kernels
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #layer 5 has 512 kernels
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),

            #layer 6 has 1024 kernels
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #fully connected layer
        self.fully_connected_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(30*30*1024, 4)

        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        return x



