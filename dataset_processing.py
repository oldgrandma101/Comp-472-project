import numpy as np
from data_cleaning import import_images
from data_cleaning import select_random_images
from data_cleaning import label_images
from data_cleaning import convert_images_to_numpy_array


from data_cleaning import angry_label
from data_cleaning import neutral_label
from data_cleaning import focused_label
from data_cleaning import happy_label


# #import final datsets to be used for project
list_final_angry=import_images("./Final_clean_dataset/final_angry")
list_final_happy=import_images("./Final_clean_dataset/final_happy")
list_final_neutral=import_images("./Final_clean_dataset/final_neutral")
list_final_focused=import_images("./Final_clean_dataset/final_focused")

# save a 25 image sample from each class to upload on moodle
select_random_images("./Final_clean_dataset/final_angry","./Samples/angry",25)
select_random_images("./Final_clean_dataset/final_happy","./Samples/happy",25)
select_random_images("./Final_clean_dataset/final_neutral","./Samples/neutral",25)
select_random_images("./Final_clean_dataset/final_focused","./Samples/focused",25)

# #lists of lables for each emotion
list_angry_labels=label_images(list_final_angry,angry_label)
list_happy_labels=label_images(list_final_happy,happy_label)
list_neutral_labels=label_images(list_final_neutral,neutral_label)
list_focused_labels=label_images(list_final_focused,focused_label)


# #convert images in lists to numpy array, they're still stored in a list
angry_images_as_array = convert_images_to_numpy_array(list_final_angry) #these are lists that contain arrays
happy_images_as_array = convert_images_to_numpy_array(list_final_happy)
neutral_images_as_array = convert_images_to_numpy_array(list_final_neutral)
focused_images_as_array = convert_images_to_numpy_array(list_final_focused)

# #add all lists together to make one big list
final_list_of_all_images = angry_images_as_array + happy_images_as_array + neutral_images_as_array+focused_images_as_array
final_list_of_all_labels = list_angry_labels + list_happy_labels + list_neutral_labels+list_focused_labels


# #************************************
# #************************************
# #Caspar and Jonny, I think you guys are going to need the numpy arrays that I made below this comment
# #in your data_visualization.py file
#
# #these are numpy arrays of each emotion that store each image as a numpy array which stores the pixel values

#************************************
#************************************
#Caspar and Jonny, I think you guys are going to need the numpy arays that I made below this comment
#in your data_visualization.py file

#these are numpy arrays of each emotion that store each image as a numpy array which stores the pixel values

array_of_angry = np.array(angry_images_as_array)
array_of_happy = np.array(happy_images_as_array)
array_of_neutral = np.array(neutral_images_as_array)
array_of_focused = np.array(focused_images_as_array)

# #numpyarrays storing labels for each emotion
array_angry_labels=np.array(list_angry_labels)
array_happy_labels=np.array(list_happy_labels)
array_neutral_labels=np.array(list_neutral_labels)
array_focused_labels = np.array(list_focused_labels)

# #numpy array containing whole dataset together and all the labels together
final_array_of_all_images = np.array(final_list_of_all_images)
final_array_of_all_labels = np.array(final_list_of_all_labels)

# added for pixel intensity diagrams below - johnny
# feel free to move these somewhere else but to keep data_visualization running quickly I moved them here

# create sample class arrays
angry_samples = import_images("./Samples/angry")
happy_samples = import_images("./Samples/happy")
focused_samples = import_images("./Samples/focused")
neutral_samples = import_images("./Samples/neutral")

# convert to numpy arrays
angry_samples_as_array = convert_images_to_numpy_array(angry_samples)
happy_samples_as_array = convert_images_to_numpy_array(happy_samples)
neutral_samples_as_array = convert_images_to_numpy_array(neutral_samples)
focused_samples_as_array = convert_images_to_numpy_array(focused_samples)

# create dirty image arrays

angry_dirty = import_images("./Final_dirty_dataset/dirty_angry")
happy_dirty = import_images("./Final_dirty_dataset/dirty_happy")
neutral_dirty = import_images("./Final_dirty_dataset/dirty_neutral")
focused_dirty = import_images("./Final_dirty_dataset/dirty_focused")

# convert to numpy arrays

angry_dirty_as_array = convert_images_to_numpy_array(angry_dirty)
happy_dirty_as_array = convert_images_to_numpy_array(happy_dirty)
neutral_dirty_as_array = convert_images_to_numpy_array(neutral_dirty)
focused_dirty_as_array = convert_images_to_numpy_array(focused_dirty)
