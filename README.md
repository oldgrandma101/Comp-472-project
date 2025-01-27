# Comp-472-project 
Group Name: AK_6
group project for comp 472 summer 2024
Group members: Jingchao Song, Anthony Phelps, Johnny Morcos

**Project Overview:**
The project aims to develop a system that uses deep learning convolutional neural networks (CNNS) and PyTorch to analyze images of students in a classroom or online meeting setting and classify them into different states or activities. We chose four emotional categories: anger, happiness, neutral, and focused/engaged.

**Contents:**
"data_cleaning.py": This script is responsible for data cleaning and preprocessing. It can resize images, enhance brightness, and convert images to grayscale. It can also import images and select random samples. (Please don't run this file, because it'll create the new dataset.)

"dataset_processing.py": This file is to import the cleaned images, converts them into numpy arrays, and prepares the final data set for visualization. It also includes functions for visualizing class distributions and sample images.

“visualization.py”: The file defines two functions to draw the class distribution and the sample image, then combines the labels and images of all the categories together, and finally calls these two functions to generate the corresponding chart. With these visualizations, you can visually view distributions and image examples for different categories in the dataset.

"data_visualization.py": This file includes the functions responsible for calculating and plotting the pixel intensity for each class on a histogram by calling 3 different functions. These histograms depict the pixel density ranging from 0-255 for each class.

"CNN_1.py", "CNN_2.py" & "CNN_3.py": These three models are convolutional neural network for image classification tasks.The model includes convolution layer, pooling layer, fully connected layer and so on. Their function is to extract image features, reduce the size of feature map, prevent overfitting, improve stability and so on. The differences are: CNN1 has a moderate number of convolutional layers, CNN 2 has fewer convolutional layers, but the number of channels per layer is significantly increased. And CNN3 has more number of convolutional layers. 

"application.py": The function of this code is to make a classification prediction of a randomly selected image using three different convolutional neural networks. Labels and category names for randomly selected images are also printed. After the prediction is made to the image, the prediction results are printed.

"testing.py": The function of this code is to test CNN_1, CNN_2 and CNN_3: calculate the accuracy of each model on the test set. 

"training.py": Three different convolutional neural networks are trained and the best model parameters are saved.

"evaluation.py": The function of this code is to evaluate the performance of three different convolutional neural networks on the test set, calculate and display their confusion matrix and various evaluation metrics.

"best_model_CNN1.pth" and "best_model_CNN2.pth" are the saved models of each CNN respectively. "best_model_CNN3.pth" exceeded GitHubs maximum file size of 100Mb so is not in this repository. To acces "best_model_CNN3.pth" please follow the link: https://drive.google.com/file/d/1UdCiwvEESQyjomHLUZDM0JjdNR3XesRx/view?usp=drive_link

"part3_model.pth" exceeded GitHubs maximum file size of 100Mb so is not in this repository. To access "part3_model.pth" please follow the link: https://drive.google.com/file/d/18rKBdaeYMoVZsKIp2lKOYBRqXlfDvL5v/view?usp=drive_link

"bias_evaluation.py": This file assesses potential bias in AI models by analyzing how the models perform across different demographic groups.

"semi_auto_labelling.py": This code helps label images by displaying them and asking users to assign them gender and age categories.

"bug_fix.py": In semi_auto_labelling.py we have found that if the picture shows a man or a middle-aged person, entering "m" will put the picture in the same folder. So making code changes through the bug fix.py file to fix errors.

"part3_cleaning.py": This code aims to enhance and process the existing image data set, and generate new image data set to improve the diversity and robustness of model training by adjusting image size, brightness, color and other characteristics.

"kcrossfold.py": This code is designed to evaluate the CNN model through K-fold cross-validation

"updated_evaluation.py": This code is to evaluating the performance of the model, increasing the speed of evaluation by running on the GPU.

"updated_training.py": The model is trained using Gpus and includes an early stop mechanism to avoid overfitting.

**Purpose of the File:**
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

application.py:
Transforms: Preprocessing of images, such as resizing. 
predict_class_of_image: Load the image dataset and tag the images.

testing.py:
model_testing: Load the specified pre-trained model and use the test set for evaluation

training.py:
Transforms: Preprocessing of images, such as resizing. 
training: Train the model, calculate the training and validation losses, 
and save the model with the least validation losses.

evaluation.py: 
evaluate_model: evaluates the model and calculates various performance values

bias_evaluation.py:
get_all_filenames_in_folder(folder_path): Recursively retrieves all filenames in the specified folder
evaluate_model(model, data_loader, target_filenames): Evaluates the model on the specified data loader and the target file name. Calculate various performance metrics such as confusion matrix, accuracy, precision, recall, and F1-scores.

semi_auto_labelling.py:
Three gender categories were used: male (m), female (f), other (o), and three age categories: young (y), middle-aged (m), and elderly (s). And users can modify the emotion variable in the script to set the "emotion" category that is currently being processed. Then, based on user input, save the image to the appropriate destination folder.

bug_fix.py:
Updated the code in semi_auto_labelling.py to fix that if the picture shows a middle-aged person or a man, the user will type 'm', causing the system to put the picture showing a middle-aged person or a man in the same folder.

part3_cleaning.py:
The script processes image folders of different categories and moods in turn, and importing, processing, and exporting enhanced images.
rename_images(folder_path): Renames all image files in the specified folder
import_images(folder_path): Import all image files in the specified folder and return a list of image objects
clean_images(list_of_dirty_pictures): Image processing, including resizing, increasing brightness, and converting to grayscale images

kcrossfold.py: Ensuring the robustness and stability of the model under different data splitting and evaluating the model performance in 10-fold cross-validation.

updated_evaluation.py: Calculating multiple performance metrics to fully evaluate the model's performance.

updated_evaluation.py: Calculating the loss and accuracy on the training set and verification set, and output the information.

**Excuting the code:**
Data Cleaning
1. Clone the repository
for example: git clone the link of repository

2. install dependencies
for example: pip install -r requirements.txt

3. Run the code

Data Visualization
1. Run the code of dataset_processing.py
   (import the clean iamges, and then covert them to numpy arrarys. And generate visualizations for class distribution and sample images.)

Classification prediction of images
1. Data Preprocessing and Loading
2. Display Class of Demo Image
3. Predict Image Class

Training three models
1. Data Preprocessing and Loading
2. Define Training Function
3. Train Models

Testing three models
1. Import PyTorch, the test data loader, and models
2. Define Model Testing Function
3. Test models

Evaluation
1. Load Pre-trained Models
2. Define Evaluation Function (evaluate_model)
3. Evaluate Models

Biased evaluation
1. Prepare the data and organize the data sets into the corresponding folders for each demographic group
2. Run the file for bias assessment

Labelling images
1. Excuting the code
2. Starting the labeling process. The script will display each image in turn and prompt the user to enter a gender and age category.

Fix the code in semi auto labelling.py
1. Excuting the code, and the script will display each image in turn
2. If the picture shows a non-middle-aged person, the user needs to enter 'q' to skip, if the picture shows a middle-aged person, the user needs to enter m or f according to the gender of the person, and then press 'Enter' to enter 'z', then the system will put the picture in a new folder.

Data cleaning and enhancement
1. Importing and process images
2. Exporting the enhanced image, and converting the image to RGB format

K-fold Cross-validation
1. Setting up hyperparameters and data transformations. Importing data sets and initializing K-fold cross-validation
2. Training the model and evaluating performance using test sets
3. Printing the mean and standard deviation of each evaluation measure

Updated model evaluation
1. Preparing data to make sure the test data set is loaded into the data loader, and the model is loaded and ready for evaluation.
2. Excuting the code evaluate the model.

Updated model training
1. Preparing data to make sure the training and validation datasets are loaded into the data loader, and the model is initialized and ready for training.
2. Call the 'training' function to train the model and monitor its performance on the training and validation sets. 

**Dependencies:**

Language: Python

matplotlib,
numpy,
PIL,
shutil,
PyTorch,
torchvision,
scikit-learn
