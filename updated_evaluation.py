import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, data_loader, device):
    # copy pasted from evaluation.py and edited to include device which can allow it to run on the GPU if connected
    # nothing else changed except removing the display of confusion matrix

    model.eval()  # change model to evaluation mode

    all_preds = []
    all_labels = []

    with torch.no_grad():  # disable gradient calculation since this is evaluation
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # added this line to send the images/labels to the selected device
            outputs = model(images)  # forward pass to get preds
            _, preds = torch.max(outputs.data, 1)  # get pred labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())  # added .cpu() to move preds & labels to CPU before beginning math

    # calculate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    # removed display of confusion matrix in this function, moved to its own thing

    # calculate the accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    #calculate the macro values
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    #calculate the micro values
    micro_precision = precision_score(all_labels, all_preds, average='micro')
    micro_recall = recall_score(all_labels, all_preds, average='micro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')

    return conf_matrix, accuracy, macro_recall, macro_precision, macro_f1, micro_precision, micro_recall, micro_f1
