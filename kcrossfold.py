# torch imports

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

# kfold imports

from sklearn.model_selection import KFold
import numpy as np

# external file imports

import CNN_3
import CNN_2
import CNN_1
from training import ImageFolderWithPaths
from updated_training import training
from updated_evaluation import evaluate_model

# GPU acceleration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use the GPU if it is available for faster runtime (from ChatGPT)

# training hyperparameters

epochs = 20
lr = 0.0001
batch = 32  # same as our normal training
kfolds = 10  # requested 10 splits for the k-fold cross validation

# use the same transform as the training file as well (taken from ChatGPT in project 1)
# resize images to 128x128, transform them to tensors and then normalize those tensors
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# import dataset & apply transform
dataset = ImageFolderWithPaths(root="./Final_clean_dataset", transform=transform)

# kfold initialization, 10 splits, shuffled to avoid manual, static split of the data

kf = KFold(n_splits=kfolds, shuffle=True, random_state=85)  # random_state saves the way we split the data for it to split it the same way everytime we run

# initialize empty arrays to store the values returned by the evaluation of each model

# model 1 values

model1_confs = []  # confusion matrices
model1_accuracies = []
model1_macro_recalls = []
model1_macro_precisions = []
model1_macro_f1s = []
model1_micro_recalls = []
model1_micro_precisions = []
model1_micro_f1s = []

# model 2 values

model2_confs = []  # confusion matrices
model2_accuracies = []
model2_macro_recalls = []
model2_macro_precisions = []
model2_macro_f1s = []
model2_micro_recalls = []
model2_micro_precisions = []
model2_micro_f1s = []

# model 3 values

model3_confs = []  # confusion matrices
model3_accuracies = []
model3_macro_recalls = []
model3_macro_precisions = []
model3_macro_f1s = []
model3_micro_recalls = []
model3_micro_precisions = []
model3_micro_f1s = []

# kfold implementation

for fold, (train_fold, test_fold) in enumerate(kf.split(dataset)):
    # use kf.split(dataset.imgs, dataset.targets) over kf.split(datasets) if you use stratified kfold for no bias
    # our data is evenly split across all classes therefore we do not need to use stratified kfold and a generic
    # dataset split is serviceable
    # TODO: If imported dataset is biased, convert kfold to stratified kfold to eliminate bias when training/testing

    print(f"Fold {fold+1} out of {kfolds}\n")

    train_data = Subset(dataset, train_fold)  # take a subset of the dataset at the fold index for training
    test_data = Subset(dataset, test_fold)  # take a subset of the dataset at the fold index for testing

    # note: train_data contains both training data & validation data, we do not need to further split for validation

    # 85% 15% split for training dataset & validation dataset
    len_train = int(0.85*len(train_data))
    len_val = len(train_data) - len_train  # if I don't do this it crashes

    # split the data into our subsets using the lengths we just calculated
    # increment the seed implemented by torch.Generator() every iteration so we don't get the exact same split of
    # the data for every fold, but still getting reproducible results
    # do not remove the + fold else the entire purpose of cross validation is lost
    train_data_subset, val_data_subset = random_split(train_data, [len_train, len_val], generator=torch.Generator().manual_seed(10+fold))

    # copied from training.py just changed the variable names
    train_loader = DataLoader(train_data_subset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_data_subset, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)

    # model, optimizer, criterion initializer, taken from training.py
    # reinitialize each fold because we do not want to use trained models over and over, it's pointless

    criterion = torch.nn.CrossEntropyLoss()  # all the models can share the same criterion

    # each model gets its own optimizer
    # send the model to the device to enable GPU acceleration

    model_1 = CNN_1.CNN_1().to(device)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr)

    model_2 = CNN_2.CNN_2().to(device)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=lr)

    model_3 = CNN_3.CNN_3().to(device)
    optimizer_3 = torch.optim.Adam(model_3.parameters(), lr=lr)

    # training each model using the training data set

    training(model_1, "CNN_1", train_loader, val_loader, criterion, optimizer_1, epochs, device)
    training(model_2, "CNN_2", train_loader, val_loader, criterion, optimizer_2, epochs, device)
    training(model_3, "CNN_3", train_loader, val_loader, criterion, optimizer_3, epochs, device)

    # evaluate each model's performance using the test data set

    # model 1 evaluation

    conf_matrix, accuracy, macro_recall, macro_precision, macro_f1, micro_precision, micro_recall, micro_f1 = evaluate_model(model_1, test_loader, device)
    model1_confs.append(conf_matrix)
    model1_accuracies.append(accuracy)
    model1_macro_recalls.append(macro_recall)
    model1_macro_precisions.append(macro_precision)
    model1_macro_f1s.append(macro_f1)
    model1_micro_recalls.append(micro_recall)
    model1_micro_precisions.append(micro_precision)
    model1_micro_f1s.append(micro_f1)

    # model 2 evaluation

    conf_matrix, accuracy, macro_recall, macro_precision, macro_f1, micro_precision, micro_recall, micro_f1 = evaluate_model(model_2, test_loader, device)
    model2_confs.append(conf_matrix)
    model2_accuracies.append(accuracy)
    model2_macro_recalls.append(macro_recall)
    model2_macro_precisions.append(macro_precision)
    model2_macro_f1s.append(macro_f1)
    model2_micro_recalls.append(micro_recall)
    model2_micro_precisions.append(micro_precision)
    model2_micro_f1s.append(micro_f1)

    # model 3 evaluation

    conf_matrix, accuracy, macro_recall, macro_precision, macro_f1, micro_precision, micro_recall, micro_f1 = evaluate_model(model_3, test_loader, device)
    model3_confs.append(conf_matrix)
    model3_accuracies.append(accuracy)
    model3_macro_recalls.append(macro_recall)
    model3_macro_precisions.append(macro_precision)
    model3_macro_f1s.append(macro_f1)
    model3_micro_recalls.append(micro_recall)
    model3_micro_precisions.append(micro_precision)
    model3_micro_f1s.append(micro_f1)

# print results

print("\nEach value in the array represents folds 1 through 10 respectively.\n")
print("\nCNN_1 Evaluation:\n")
print(f"Accuracy: {np.mean(model1_accuracies)} ± {np.std(model1_accuracies)}")
print(f"Macro Recall: {np.mean(model1_macro_recalls)} ± {np.std(model1_macro_recalls)}")
print(f"Macro Precision: {np.mean(model1_macro_precisions)} ± {np.std(model1_macro_precisions)}")
print(f"Macro F1: {np.mean(model1_macro_f1s)} ± {np.std(model1_macro_f1s)}")
print(f"Macro Recall: {np.mean(model1_micro_recalls)} ± {np.std(model1_micro_recalls)}")
print(f"Macro Precision: {np.mean(model1_micro_precisions)} ± {np.std(model1_micro_precisions)}")
print(f"Macro F1: {np.mean(model1_micro_f1s)} ± {np.std(model1_micro_f1s)}")

print("\nCNN_2 Evaluation:\n")
print(f"Accuracy: {np.mean(model2_accuracies)} ± {np.std(model2_accuracies)}")
print(f"Macro Recall: {np.mean(model2_macro_recalls)} ± {np.std(model2_macro_recalls)}")
print(f"Macro Precision: {np.mean(model2_macro_precisions)} ± {np.std(model2_macro_precisions)}")
print(f"Macro F1: {np.mean(model2_macro_f1s)} ± {np.std(model2_macro_f1s)}")
print(f"Macro Recall: {np.mean(model2_micro_recalls)} ± {np.std(model2_micro_recalls)}")
print(f"Macro Precision: {np.mean(model2_micro_precisions)} ± {np.std(model2_micro_precisions)}")
print(f"Macro F1: {np.mean(model2_micro_f1s)} ± {np.std(model2_micro_f1s)}")

print("\nCNN_3 Evaluation:\n")
print(f"Accuracy: {np.mean(model3_accuracies)} ± {np.std(model3_accuracies)}")
print(f"Macro Recall: {np.mean(model3_macro_recalls)} ± {np.std(model3_macro_recalls)}")
print(f"Macro Precision: {np.mean(model3_macro_precisions)} ± {np.std(model3_macro_precisions)}")
print(f"Macro F1: {np.mean(model3_macro_f1s)} ± {np.std(model3_macro_f1s)}")
print(f"Macro Recall: {np.mean(model3_micro_recalls)} ± {np.std(model3_micro_recalls)}")
print(f"Macro Precision: {np.mean(model3_micro_precisions)} ± {np.std(model3_micro_precisions)}")
print(f"Macro F1: {np.mean(model3_micro_f1s)} ± {np.std(model3_micro_f1s)}")
