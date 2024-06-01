import numpy as np
import matplotlib.pyplot as plt
from dataset_processing import angry_images_as_array, happy_images_as_array, neutral_images_as_array, focused_images_as_array, angry_samples_as_array, happy_samples_as_array, neutral_samples_as_array, focused_samples_as_array, angry_dirty_as_array, happy_dirty_as_array, neutral_dirty_as_array, focused_dirty_as_array

# import the class arrays instead of the combined array since we need pixel intensity by class

def pixel_intensity_calculation(image_array, class_name):
    # Calculate the pixel intensity of a certain class of images (angry, happy, etc...),
    # the array of images itself, and then take this information and plot it on a histogram with matplotlib
    # this works for the sample images too
    pixel_intensity = np.concatenate([image.flatten() for image in image_array])  # flatten the pixel information for each image into a 1D array and then concatenate them all together
    pixel_intensity_plot(pixel_intensity, class_name)

def pixel_intensity_plot(pixel_intensity, label_info):
    # Plot pixel intensity on a histogram, reusable for both class & sample calculations
    plt.hist(pixel_intensity, bins=256, label=label_info)  # idea for bins=256 came from chat-gpt; since colors go from 0-255 (hexadecimal)
    plt.xlabel('Intensity')
    plt.ylabel('Freq.')  # Y-axis shows the frequency of each color's occurrence in the images
    plt.legend()
    plt.show()


def plot_arrays():
    # function to plot the histograms for all 8 arrays, update with any new arrays that need to be plotted

    # plotting the clean classes

    pixel_intensity_calculation(angry_images_as_array, "Angry")
    pixel_intensity_calculation(happy_images_as_array, "Happy")
    pixel_intensity_calculation(focused_images_as_array, "Focused")
    pixel_intensity_calculation(neutral_images_as_array, "Neutral")

    # plot the sample classes

    pixel_intensity_calculation(angry_samples_as_array, "Angry Samples")
    pixel_intensity_calculation(happy_samples_as_array, "Happy Samples")
    pixel_intensity_calculation(focused_samples_as_array, "Focused Samples")
    pixel_intensity_calculation(neutral_samples_as_array, "Neutral Samples")

    # plot the dirty classes

    pixel_intensity_calculation(angry_dirty_as_array, "Angry Dirty")
    pixel_intensity_calculation(happy_dirty_as_array, "Happy Dirty")
    pixel_intensity_calculation(focused_dirty_as_array, "Focused Dirty")
    pixel_intensity_calculation(neutral_dirty_as_array, "Neutral Dirty")


# code to execute

plot_arrays()
