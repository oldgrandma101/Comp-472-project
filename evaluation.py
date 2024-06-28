import torch
import torch.nn
import CNN_1
import CNN_2
import CNN_3
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from training import test_loader
from training import  part3_test_loader





main_model = CNN_1.CNN_1()
main_model.load_state_dict(torch.load("./best_model_CNN1.pth"))

variant_model1 = CNN_2.CNN_2()
variant_model1.load_state_dict(torch.load("./best_model_CNN2.pth"))

variant_model2 = CNN_3.CNN_3()
variant_model2.load_state_dict(torch.load("./best_model_CNN3.pth"))

part3_model = CNN_3.CNN_3()
part3_model.load_state_dict(torch.load("./part3_model.pth"))



def evaluate_model(model, data_loader):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            images, labels, _ = data
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # calculate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    plt.show()

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

#####################################
#Evaluation for part2

# main_conf_matrix, main_accuracy, main_macro_recall, main_macro_precision, main_macro_f1, main_micro_precision, main_micro_recall, main_micro_f1 = evaluate_model(main_model,test_loader)
#
# print(main_conf_matrix)
#
# print("main_accuracy: ",main_accuracy)
# print("main_macro_recall: ", main_macro_recall)
# print("main_macro_precision: ",main_macro_precision)
# print("main_macro_f1",main_macro_f1)
# print("main_micro_precision: ",main_micro_precision)
# print("main_micro_recall: ",main_micro_recall)
# print("main_micro_f1: ",main_micro_f1)
#
#
# v1_conf_matrix, v1_accuracy, v1_macro_recall, v1_macro_precision, v1_macro_f1, v1_micro_precision, v1_micro_recall, v1_micro_f1 = evaluate_model(variant_model1,test_loader)
#
#
# print(v1_conf_matrix)
#
# print("v1_accuracy: ",v1_accuracy)
# print("v1_macro_recall: ", v1_macro_recall)
# print("v1_macro_precision: ",v1_macro_precision)
# print("v1_macro_f1",v1_macro_f1)
# print("v1_micro_precision: ",v1_micro_precision)
# print("v1_micro_recall: ",v1_micro_recall)
# print("v1_micro_f1: ",v1_micro_f1)
#
#
# v2_conf_matrix, v2_accuracy, v2_macro_recall, v2_macro_precision, v2_macro_f1, v2_micro_precision, v2_micro_recall, v2_micro_f1 = evaluate_model(variant_model2,test_loader)
#
# print(v2_conf_matrix)
# print("v2_accuracy: ",v2_accuracy)
# print("v2_macro_recall: ", v2_macro_recall)
# print("v2_macro_precision: ",v2_macro_precision)
# print("v2_macro_f1",v2_macro_f1)
# print("v2_micro_precision: ",v2_micro_precision)
# print("v2_micro_recall: ",v2_micro_recall)
# print("v2_micro_f1: ",v2_micro_f1)

######################################
#evaluation for part 3
part3_conf_matrix, part3_accuracy, part3_macro_recall, part3_macro_precision, part3_macro_f1, part3_micro_precision, part3_micro_recall, part3_micro_f1 = evaluate_model(part3_model,part3_test_loader)

print(part3_conf_matrix)

print("part3_accuracy: ",part3_accuracy)
print("part3_macro_recall: ", part3_macro_recall)
print("part3_macro_precision: ",part3_macro_precision)
print("part3_macro_f1",part3_macro_f1)
print("part3_micro_precision: ",part3_micro_precision)
print("part3_micro_recall: ",part3_micro_recall)
print("part3_micro_f1: ",part3_micro_f1)