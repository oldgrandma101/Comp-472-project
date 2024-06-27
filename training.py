import torch
import torch.nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import CNN_1
import CNN_2
import CNN_3
from torchvision.datasets import ImageFolder
import numpy

#extends ImageFolder so that we can process the file names too, added for part 3 of the project, used ChatGPT for this class
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)



num_epochs = 20
learning_rate = 0.0001

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # from ChatGPT

dataset = ImageFolderWithPaths(root="./Final_clean_dataset", transform=transform)  # apply the transform to our dataset and load it

# 70% - 15% - 15% split for train/val/test, distributed randomly
#train_dataset, validation_dataset, test_dataset = random_split(dataset, [int(0.7*len(dataset)), int(0.15*len(dataset)), int(0.15*len(dataset))])

# corrected above line by chatgpt

train_size = int(0.7*len(dataset))
validation_size = int(0.15*len(dataset))
test_size = len(dataset) - train_size - validation_size  # apparently there is a rounding issue when u do this any other way

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])  # split it according to our size split 70-15-15

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # load our datasets into dataloaders with 256 batch sizes

#######################################################################################
#######################################################################################
#Part 3 Dataset

part3_dataset = ImageFolderWithPaths(root="./part3_final_dataset", transform=transform)

part3_train_size = int(0.7*len(part3_dataset))
part3_validation_size = int(0.15*len(part3_dataset))
part3_test_size = len(part3_dataset)-part3_train_size-part3_validation_size

part3_train_dataset, part3_validation_dataset, part3_test_dataset = random_split(part3_dataset,[part3_train_size, part3_validation_size, part3_test_size])

part3_train_loader = DataLoader(part3_train_dataset, batch_size=32, shuffle=True)
part3_validation_loader = DataLoader(part3_validation_dataset, batch_size=32, shuffle=False)
part3_test_loader = DataLoader(part3_test_dataset, batch_size=32, shuffle=False)



def training(model, train_loader, validation_loader, criterion, optimizer, num_epochs=num_epochs, model_path="best_model.pth"):
    loss_list = []
    best_loss = float('inf')  # keeps track of the best validation loss for the model
    patience_count = 0  # compares count with patience to create early stopping

    for epoch in range(num_epochs):

        model.train()  # set model to training mode
        training_loss = 0  # reset to 0 for each epoch
        acc_list = []  # reset accuracy list to 0 for each epoch

        for i, (images, labels,_) in enumerate(train_loader):   #had to add _ because I added filepaths to the tuple when extending the ImageFolder class

            # fwd pass
            outputs = model(images)
            loss = criterion(outputs, labels)  # compute loss
            loss_list.append(loss.item())  # add the loss value to our array

            # back & optimize
            optimizer.zero_grad()  # reset gradient after every forward pass
            loss.backward()  # backwards propagation
            optimizer.step()  # update parameter for next cycle

            training_loss += loss.item()  # calculate the total loss while training for the epoch, so we can compare and choose the best one to store

            # train accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)  # use torch.max to make predictions on our dataset
            correct = (predicted == labels).sum().item()  # store all the correct ones
            acc_list.append(correct / total)  # calculate total accuracy of this passthrough


        validation_loss = 0.0  # this will keep track of our validation loss to compare best_loss to
        model.eval()  # switch model to evaluation mode

        with torch.no_grad():  # disable gradient changes for validation
            for i, (images, labels,_) in enumerate(validation_loader):  #had to add _ because I added filepaths to the tuple when extending the ImageFolder class
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()  # forward pass through the validation set to compute validation loss only without changing the model

        validation_loss = validation_loss / len(validation_loader)  # from ChatGPT

        print(f"Epoch {epoch+1}, Validation Loss: {validation_loss},  Final Passthrough Accuracy: {round(numpy.mean(acc_list) * 100, 2)}")  # print info for developer

        if validation_loss < best_loss:
            best_loss = validation_loss  # if this epoch's validation loss is smaller, then set it as the new best
            torch.save(model.state_dict(), model_path)  # if it has the best validation loss, it is the best model, so store it
            patience_count = 0  # reset patience if a new best model is found
        else:
            patience_count += 1
            if patience_count >= 6:  # patience of 6
                print('Early stopping mechanism activated to avoid over fitting of the model.')
                break


criterion = torch.nn.CrossEntropyLoss()  # all the models can share the same criterion

# model_1 = CNN_1.CNN_1()
# optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
#
# model_2 = CNN_2.CNN_2()
# optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate)
#
# model_3 = CNN_3.CNN_3()
# optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=learning_rate)  # define our 3 models with their optimizers respectively so that we can now store the best trained model for each

#the lines underneath this comment have been commented out so that the training file isn't ran everytime
#we import a function from this file in other files

# training(model_1, train_loader, validation_loader, criterion, optimizer_1, num_epochs, "best_model_CNN1.pth")
# training(model_2, train_loader, validation_loader, criterion, optimizer_2, num_epochs, "best_model_CNN2.pth")
# training(model_3, train_loader, validation_loader, criterion, optimizer_3, num_epochs, "best_model_CNN3.pth")  # execute training for each neural network

# part3_model = CNN_3.CNN_3()
# part3_optimizer = torch.optim.Adam(part3_model.parameters(), lr=learning_rate)
#
# training(part3_model, part3_train_loader, part3_validation_loader, criterion, part3_optimizer, num_epochs, "part3_model.pth")