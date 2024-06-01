import matplotlib.pyplot as plt
import numpy as np

# Visual class distribution
def plot_class_distribution(labels, class_names, title='Class Distribution'):
    plt.figure(figsize=(10, 6))
    plt.hist(labels, bins=np.arange(len(class_names) + 1) - 0.5, edgecolor='black')
    plt.xticks(range(len(class_names)), class_names)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title(title)
    plt.grid(axis='y')
    plt.show()

# Visual sample images
def plot_sample_images(images, labels, class_names, num_samples_per_class=5):
    plt.figure(figsize=(15, 10))
    for class_index, class_name in enumerate(class_names):
        class_images = [img for img, label in zip(images, labels) if label == class_index + 1]
        for i in range(num_samples_per_class):
            plt.subplot(len(class_names), num_samples_per_class, class_index * num_samples_per_class + i + 1)
            plt.imshow(class_images[i], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.ylabel(class_name)
    plt.suptitle('Sample Images from Each Class', fontsize=16)
    plt.show()

# Define a class name
class_names = ['Angry', 'Happy', 'Neutral', 'Focused']

# Using numpy arrays
final_array_of_all_labels = np.concatenate((array_angry_labels, array_happy_labels, array_neutral_labels, array_focused_labels))
final_array_of_all_images = np.concatenate((array_of_angry, array_of_happy, array_of_neutral, array_of_focused))

# Plot the class distribution
plot_class_distribution(final_array_of_all_labels, class_names)

# Draw sample image
plot_sample_images(final_array_of_all_images, final_array_of_all_labels, class_names)
