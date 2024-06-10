#this project needs 3 different variants of CNNs to experiment with, this is the first one
#This variant has 10 convolutional layers with 1 linear layer to classify into the 4 classes
#I didn't add layers that produce 512 outputs, that could be something to try if this doesn't perform well
#Can also try different activation functions instead of leakyRELU or different pooling than MaxPool2D
#can also experiment with the number of convolutional layers

import torch.nn as nn
class CNN_1(nn.Module):
    #define constructor
    def __init__(self):
        super(CNN_1, self).__init__()
        self.conv_layer = nn.Sequential(    #defines convolutional layers in CNN

            #in_channels = 1 because our images are grayscale
            #layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            #layer 2
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),


            #layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            #layer 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #layer 5
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            #layer 6
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #layer 7
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            #layer 8
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #layer 9
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            #layer 10
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        #fully conected layer
        self.fully_connected_layer = nn.Sequential(
            nn.Dropout(p=0.1),  #this makes 10% of all weights zero, Idea is to avoid becoming too reliant on certain connections and avoid overfitting
            nn.Linear(7 * 7 * 256, 4)   #4 because we have 4 emotions to classify
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        return x

