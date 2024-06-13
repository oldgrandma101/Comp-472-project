
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import CNN_1
import CNN_2
import CNN_3


transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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


def predict_class_of_image(model_class,model_path, data_loader):
    #create and load best model
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    prediction = []
    with torch.no_grad():
        for images,_ in data_loader:
            output = model(images)
            _,predicted = torch.max(output,1)   #from ChatGPT
            prediction.extend(predicted.numpy())    #from ChatGPT
    return prediction


predictions_CNN1 = predict_class_of_image(CNN_1.CNN_1, "best_model_CNN1.pth", demo_loader)
print("Predictions from CNN_1:", predictions_CNN1)
predictions_CNN2 = predict_class_of_image(CNN_2.CNN_2, "best_model_CNN2.pth", demo_loader)
print("Predictions from CNN_2:", predictions_CNN2)
predictions_CNN3 = predict_class_of_image(CNN_3.CNN_3, "best_model_CNN3.pth", demo_loader)
print("Predictions from CNN_3:", predictions_CNN3)
