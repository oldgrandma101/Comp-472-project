# Comp-472-project 
Group Name: AK_6
group project for comp 472 summer 2024
Group members: Jingchao Song, Anthony Phelps, Johnny Morcos

Project Overview:
The project aims to develop a system that uses deep learning convolutional neural networks (CNNS) and PyTorch to analyze images of students in a classroom or online meeting setting and classify them into different states or activities. We chose four emotional categories: anger, happiness, neutral, and focused/engaged.

Contents:
"data_cleaning.py": This script is responsible for data cleaning and preprocessing. It can resize images, enhance brightness, and convert images to grayscale. It can also import images and select random samples. (Please don't run this file, because it'll create the new dataset.)

"dataset_processing.py": This file is to import the cleaned images, converts them into numpy arrays, and prepares the final data set for visualization. It also includes functions for visualizing class distributions and sample images.

“visualization.py”: The file defines two functions to draw the class distribution and the sample image, then combines the labels and images of all the categories together, and finally calls these two functions to generate the corresponding chart. With these visualizations, you can visually view distributions and image examples for different categories in the dataset.

"data_visualization.py": This file includes the functions responsible for calculating and plotting the pixel intensity for each class on a histogram by calling 3 different functions. These histograms depict the pixel density ranging from 0-255 for each class.

Purpose of the File:
data_cleaning.py: 
import_images(folder_path): Imports images from the folder.
select_random_images: Randomly select a specified number of images from the source folder and save them.
clean_images(list_of_dirty_pictures): Clearing images by resizing, enhancing brightness, and converting to grayscale.
label_images: Assigns labels to the images.

dataset_processing.py:
Imports cleaned datasets from the final clean dataset folder.
label_images: Creates label lists for each emotion.
Numpy arrays: Stores images and labels as numpy arrays.
plot_class_distribution: Plots clss distribution using matplotlib
plot_sample_images: Plots sample images for each class.

Excuting the code:
First: Data Cleaning
1. Clone the repository
for example: git clone the link of repository

2. install dependencies
for example: pip install -r requirements.txt

3. Run the code

Second: Data Visualization
1. Run the code of dataset_processing.py
   (import the clean iamges, and then covert them to numpy arrarys. And generate visualizations for class distribution and sample images.)

Dependencies:

Language: Python

matplotlib
numpy
PIL
shutil

