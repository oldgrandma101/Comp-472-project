import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# define an evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # calculate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # calculate the performance index
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return conf_matrix, accuracy, precision, recall, f1

# Define the function to draw the confusion matrix
def plot_confusion_matrix(conf_matrix):
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

#Define the main model and variant model
model_main = CNN_1()
model_variant1 = CNN_2()
model_variant2 = CNN_3()

# load the trained models' parameters
model_main.load_state_dict(torch.load(''))
model_variant1.load_state_dict(torch.load(''))
model_variant2.load_state_dict(torch.load(''))

# evaluate each model
conf_matrix_main, accuracy_main, precision_main, recall_main, f1_main = evaluate_model(model_main, test_loader)
conf_matrix_variant1, accuracy_variant1, precision_variant1, recall_variant1, f1_variant1 = evaluate_model(model_variant1, test_loader)
conf_matrix_variant2, accuracy_variant2, precision_variant2, recall_variant2, f1_variant2 = evaluate_model(model_variant2, test_loader)

# display the index
print(f'Main Model - Accuracy: {accuracy_main}, Precision: {precision_main}, Recall: {recall_main}, F1 Score: {f1_main}')
print(f'Variant 1 - Accuracy: {accuracy_variant1}, Precision: {precision_variant1}, Recall: {recall_variant1}, F1 Score: {f1_variant1}')
print(f'Variant 2 - Accuracy: {accuracy_variant2}, Precision: {precision_variant2}, Recall: {recall_variant2}, F1 Score: {f1_variant2}')

# draw the matrix
plot_confusion_matrix(conf_matrix_main)
plot_confusion_matrix(conf_matrix_variant1)
plot_confusion_matrix(conf_matrix_variant2)

# generate the table
data = {
    'Model': ['Main Model', 'Variant 1', 'Variant 2'],
    'Accuracy': [accuracy_main, accuracy_variant1, accuracy_variant2],
    'Precision': [precision_main, precision_variant1, precision_variant2],
    'Recall': [recall_main, recall_variant1, recall_variant2],
    'F1 Score': [f1_main, f1_variant1, f1_variant2]
}

df = pd.DataFrame(data)
print(df)


