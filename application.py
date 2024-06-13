import os
import random
import shutil
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import CNN_1
import CNN_2
import CNN_3


transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # from ChatGPT

#ImageFolder automatically labels all images based on folder structure
demo_dataset = datasets.ImageFolder(root="./Final_clean_dataset", transform=transform)  # applied same transform that is applied before training

#only want 1 image for the demo
demo_size = 1
the_rest_size = len(demo_dataset)-1

demo_image, the_rest_of_images = random_split(demo_dataset,[demo_size,the_rest_size])

demo_loader = DataLoader(demo_image, batch_size=1, shuffle=False)
the_rest_loader = DataLoader(the_rest_of_images, batch_size=32,shuffle=True)

#show the class of the randomly selcted demo_image

#line 29-33 is from ChatGPT
class_to_idx = demo_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
for images, labels in demo_loader:
    label = labels.item()
    class_name = idx_to_class[label]
    print("The label and class name of the image randomly selcted for demo: ")
    print("Label: ",label, "    Class Name: ", class_name)

#I need to load CNN models now and predict class of demo_image

