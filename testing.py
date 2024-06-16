import torch
from training import test_loader
import CNN_1
import CNN_2
import CNN_3


def model_testing(neuralnetwork, path, modelName):
    model = neuralnetwork()  # load an instance of the passed neural network
    model.load_state_dict(torch.load(path))  # load the pre-trained model from the path passed as an argument
    model.eval()  # set model to evaluation mode

    with torch.no_grad():  # this prevents further training of the models
        total, correct = 0, 0
        for (i, (image, label)) in enumerate(test_loader):
            out = model(image)  # gets the predictions and saves them to out to call later with torch.max
            _, predicted = torch.max(out.data,
                                     1)  # from ChatGPT, _ is used because the first set of data is not the information we need
            total = total + label.size(0)  # this is the total amount of labels, size(0) returns that (number of data entries)
            correct = correct + (
                    predicted == label).sum().item()  # compares the predicted value to our label, if it is correct, add 1 to correct

    if total != 0:  # avoid divide by zero error
        accuracy = correct / total
        print(f"Test accuracy of {modelName}: {accuracy * 100: .2f}%")
    else:
        print(f"Test set empty, reload data and try again.")


model_testing(CNN_1.CNN_1, "./best_model_CNN1.pth", "CNN1")

model_testing(CNN_2.CNN_2, "./best_model_CNN2.pth", "CNN2")

model_testing(CNN_3.CNN_3, "./best_model_CNN3.pth", "CNN3")
