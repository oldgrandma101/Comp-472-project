import matplotlib.pyplot as plt
import numpy as np
from dataset_processing import final_array_of_all_images, final_array_of_all_labels

# python is 0-indexed, so we need to change the numbers for all the labels and move them down by one so our histogram
# is created accurately
corrected_all_labels = [label - 1 for label in final_array_of_all_labels]  # chatGPT was used for this line

# Visual class distribution
def plot_class_distribution(labels, class_names, title='Class Distribution'):
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=np.arange(len(class_names) + 1) - 0.5, edgecolor='black')
    plt.xticks(range(len(class_names)), class_names)  # chatGPT was used for this line
    plt.xlabel('Class')
    plt.ylabel('Number of Images')  # chatGPT was used for this line
    plt.title(title)
    plt.grid(axis='y')  # chatGPT was used for this line
    plt.show()

# Visual sample image
def plot_sample_images(images, labels, class_names, numsamples_per_class=5):
    plt.figure(figsize=(15, 10))
    for class_index, class_name in enumerate(class_names):
        class_images = [img for img, label in zip(images, labels) if label == class_index + 1]  # chatGPT was used for this line

        for i in range(numsamples_per_class):
            plt.subplot(len(class_names), numsamples_per_class, class_index * numsamples_per_class + i + 1)

            image = class_images[i]
            if len(image.shape) == 2:  # grayscale
                plt.imshow(image, cmap='gray')  # chatGPT was used for this line
            if len(image.shape) == 3:  # RGB
                plt.imshow(image)

            plt.axis('off')  # chatGPT was used for this line
            if i == 0:
                plt.ylabel(class_name)

    plt.suptitle('Sample Images from Each Class', fontsize=16)  # chatGPT was used for this line
    plt.show()


# Define a class name 'Angry', 'Happy', 'Neutral', 'Focused'
class_names = ['Angry', 'Happy', 'Neutral', 'Focused']

# Plot the class distribution
plot_class_distribution(corrected_all_labels, class_names)

# Sample image
plot_sample_images(final_array_of_all_images, final_array_of_all_labels, class_names)

