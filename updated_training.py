import torch
import numpy


def training(model, model_name, train_loader, validation_loader, criterion, optimizer, num_epochs, device):
    # copied, pasted and edited from training.py
    # only changes include adding device & related lines (to(device)) and adjusting the printing of the training
    # accuracy for each epoch (accounted for batches unlike project 1)

    print(f"Training {model_name}: \n")

    loss_list = []
    best_loss = float('inf')  # keeps track of the best validation loss for the model
    patience_count = 0  # compares count with patience to create early stopping

    for epoch in range(num_epochs):

        model.train()  # set model to training mode
        training_loss = 0  # reset to 0 for each epoch
        acc_list = []  # reset accuracy list to 0 for each epoch

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # send images/labels to selected device

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
            _, predicted = torch.max(outputs, 1)  # use torch.max to make predictions on our dataset
            correct = (predicted == labels).sum().item()  # store all the correct ones
            acc_list.append(correct / total)  # calculate total accuracy of this passthrough

        validation_loss = 0.0  # this will keep track of our validation loss to compare best_loss to
        model.eval()  # switch model to evaluation mode

        with torch.no_grad():  # disable gradient changes for validation
            for i, (images, labels) in enumerate(validation_loader):
                images, labels = images.to(device), labels.to(device)  # send images/labels to selected device
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()  # forward pass through the validation set to compute validation loss only without changing the model

        validation_loss /= len(validation_loader)  # from ChatGPT

        # correct training accuracy display from ChatGPT
        print(
            f"Epoch {epoch + 1}, Validation Loss: {validation_loss}, Training Accuracy: {round(numpy.mean(acc_list) * 100, 2)}%")  # print info for developer

        if validation_loss < best_loss:  # even if we aren't saving the best model, we still need to store the best_loss in order to trigger patience
            best_loss = validation_loss  # if this epoch's validation loss is smaller, then set it as the new best
            # removed model saving, we don't need to save the best model this time
            patience_count = 0  # reset patience if a new best model is found
        else:
            patience_count += 1
            if patience_count >= 6:  # patience of 6
                print('Early stopping mechanism activated to avoid over fitting of the model.')
                break
