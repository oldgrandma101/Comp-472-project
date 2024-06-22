import torch
import torch.nn
import CNN_3
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from training import test_loader
import os


# changed from CNN_1 in part 2 to CNN_3 in part 3 becuase it performed better
main_model = CNN_3.CNN_3()
main_model.load_state_dict(torch.load("./best_model_CNN3.pth"))


def get_all_filenames_in_folder(folder_path):   #ChatGPT was used to help create this function
    all_filenames = set()
    for root, _, files in os.walk(folder_path):
        for file in files:
            all_filenames.add(file)
    return all_filenames


def evaluate_model(model, data_loader, target_filenames):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in data_loader:
            images, labels, paths = data
            filenames = [os.path.basename(path) for path in paths]  #ChatGPT was used to help make this line

            filtered_indices = [i for i, filename in enumerate(filenames) if filename in target_filenames]  #ChatGPT was used to help make this line
            if not filtered_indices:#ChatGPT was used to help make this line
                continue

            filtered_images = images[filtered_indices]  #ChatGPT was used to help make this line
            filtered_labels = labels[filtered_indices]  #ChatGPT was used to help make this line

            if len(filtered_images) > 0:
                outputs = model(filtered_images)
                _, preds = torch.max(outputs.data, 1)
                all_preds.extend(preds.numpy())
                all_labels.extend(filtered_labels.numpy())

    # calculate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    plt.show()

    # calculate the accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    # calculate the macro values
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    # calculate the micro values
    micro_precision = precision_score(all_labels, all_preds, average='micro')
    micro_recall = recall_score(all_labels, all_preds, average='micro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')

    return conf_matrix, accuracy, macro_recall, macro_precision, macro_f1, micro_precision, micro_recall, micro_f1


# # evaluate model on part3_young
young_target_filenames = get_all_filenames_in_folder("./part3_young")
young_conf_matrix, young_accuracy, young_macro_recall, young_macro_precision, young_macro_f1, young_micro_precision, young_micro_recall, young_micro_f1 = evaluate_model(main_model, test_loader, young_target_filenames)

print(young_conf_matrix)

print("young_accuracy: ", young_accuracy)
print("young_macro_recall: ", young_macro_recall)
print("young_macro_precision: ", young_macro_precision)
print("young_macro_f1", young_macro_f1)
print("young_micro_precision: ", young_micro_precision)
print("young_micro_recall: ", young_micro_recall)
print("young_micro_f1: ", young_micro_f1)


# # evaluate model on part3_middleaged
middleaged_target_filenames = get_all_filenames_in_folder("./part3_middleaged")
middleaged_conf_matrix, middleaged_accuracy, middleaged_macro_recall, middleaged_macro_precision, middleaged_macro_f1, middleaged_micro_precision, middleaged_micro_recall, middleaged_micro_f1 = evaluate_model(main_model, test_loader, middleaged_target_filenames)

print(middleaged_conf_matrix)

print("middleaged_accuracy: ", middleaged_accuracy)
print("middleaged_macro_recall: ", middleaged_macro_recall)
print("middleaged_macro_precision: ", middleaged_macro_precision)
print("middleaged_macro_f1", middleaged_macro_f1)
print("middleaged_micro_precision: ", middleaged_micro_precision)
print("middleaged_micro_recall: ", middleaged_micro_recall)
print("middleaged_micro_f1: ", middleaged_micro_f1)

# eva;uate model on part3_senior
senior_target_filenames = get_all_filenames_in_folder("./part3_senior")
senior_conf_matrix, senior_accuracy, senior_macro_recall, senior_macro_precision, senior_macro_f1, senior_micro_precision, senior_micro_recall, senior_micro_f1 = evaluate_model(main_model, test_loader, senior_target_filenames)

print(senior_conf_matrix)

print("senior_accuracy: ", senior_accuracy)
print("senior_macro_recall: ", senior_macro_recall)
print("senior_macro_precision: ", senior_macro_precision)
print("senior_macro_f1", senior_macro_f1)
print("senior_micro_precision: ",senior_micro_precision)
print("senior_micro_recall: ", senior_micro_recall)
print("senior_micro_f1: ", senior_micro_f1)

#evaluate model on part3_male
male_target_filenames = get_all_filenames_in_folder("./part3_male")
male_conf_matrix, male_accuracy, male_macro_recall, male_macro_precision, male_macro_f1, male_micro_precision, male_micro_recall, male_micro_f1 = evaluate_model(main_model, test_loader, male_target_filenames)

print(male_conf_matrix)

print("male_accuracy: ", male_accuracy)
print("male_macro_recall: ", male_macro_recall)
print("male_macro_precision: ", male_macro_precision)
print("male_macro_f1", male_macro_f1)
print("male_micro_precision: ", male_micro_precision)
print("male_micro_recall: ", male_micro_recall)
print("male_micro_f1: ", male_micro_f1)

#evaluate model on part3_female
female_target_filenames = get_all_filenames_in_folder("./part3_female")
female_conf_matrix, female_accuracy, female_macro_recall, female_macro_precision, female_macro_f1, female_micro_precision, female_micro_recall, female_micro_f1 = evaluate_model(main_model, test_loader, female_target_filenames)

print(female_conf_matrix)

print("female_accuracy: ", female_accuracy)
print("female_macro_recall: ", female_macro_recall)
print("female_macro_precision: ", female_macro_precision)
print("female_macro_f1", female_macro_f1)
print("female_micro_precision: ", female_micro_precision)
print("female_micro_recall: ", female_micro_recall)
print("female_micro_f1: ", female_micro_f1)

#evaluate model on part3_other
other_target_filenames = get_all_filenames_in_folder("./part3_other")
other_conf_matrix, other_accuracy, other_macro_recall, other_macro_precision, other_macro_f1, other_micro_precision, other_micro_recall, other_micro_f1 = evaluate_model(main_model, test_loader, other_target_filenames)

print(other_conf_matrix)

print("other_accuracy: ", other_accuracy)
print("other_macro_recall: ", other_macro_recall)
print("other_macro_precision: ", other_macro_precision)
print("other_macro_f1", other_macro_f1)
print("other_micro_precision: ", other_micro_precision)
print("other_micro_recall: ", other_micro_recall)
print("other_micro_f1: ", other_micro_f1)


print("the script is finished")