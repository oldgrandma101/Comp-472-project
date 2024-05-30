from PIL import Image,ImageEnhance
import os
import numpy as np

new_size=(244,244)
new_brightness=1.2
angry_label=1
neutral_label=2
focused_label=3
happy_label=4
image_counter=0

def import_images(folder_path):
    original_images = []
    import_counter=0
    for filename in os.listdir(folder_path):    #used chatgpt, I have to cite it
        if (import_counter==500):
            return original_images
        if filename.endswith('.jpg'):
            with Image.open(os.path.join(folder_path, filename)) as img:
                original_images.append(img.copy())
                import_counter+=1

    return original_images

def clean_images(list_of_dirty_pictures):
    clean_pictures=[]
    for img in list_of_dirty_pictures:
        resized_image = img.resize(new_size)
        enhancer = ImageEnhance.Brightness(resized_image)
        brighter_image = enhancer.enhance(new_brightness)
        greyscale_image = brighter_image.convert("L")  # Convert to greyscale
        clean_pictures.append(greyscale_image)

    return clean_pictures


def export_images(new_folder_path, list_of_images_to_export):
    os.makedirs(new_folder_path, exist_ok=True) #used chatgpt, I have to cite it
    global image_counter
    for img in list_of_images_to_export:
        output_path = os.path.join(new_folder_path, f"new__image_{image_counter}.jpg")
        img.save(output_path)
        image_counter+=1

    return

def label_images(list_of_pics_to_label, label_to_assign):
    list_of_labels=[]

    for x in list_of_pics_to_label:
        list_of_labels.append(label_to_assign)

    return list_of_labels

def convert_images_to_numpy_array(list_of_images_to_convert):
    new_list=[]
    for img in list_of_images_to_convert:
        x=np.array(img)
        new_list.append(x)
    return new_list

#***********************************
#***********************************
#import and choose 500 images from fer datasets, focused dataset, and import group pictures
fer_happy=import_images("./happy")
fer_angry=import_images("./angry")
fer_neutral=import_images("./neutral")
#focused = import_images("./focused")       don't have this data set yet
group_happy=import_images("./group_happy")
group_angry=import_images("./group_angry")
group_neutral=import_images("./group_neutral")
group_engaged=import_images("./group_engaged")

#***************************************
#***************************************
#make list of clean images for each class
clean_happy=clean_images(fer_happy)
clean_angry=clean_images(fer_angry)
clean_neutral=clean_images(fer_neutral)
#clean_focused=clean_images(focused)    don't have this data set yet
clean_group_happy=clean_images(group_happy)
clean_group_angry=clean_images(group_angry)
clean_group_neutral=clean_images(group_neutral)
clean_group_focused=clean_images(group_engaged)


#Add group pics to appropriate datasets
clean_group_happy += clean_happy
clean_group_angry +=clean_angry
clean_group_neutral +=clean_neutral
#clean_group_focused +=clean_focused    missing dataset

#save the images in the list to save dataset for project
export_images("./final_happy",clean_group_happy)
export_images("./final_angry",clean_group_angry)
export_images("./final_neutral",clean_group_neutral)
#export_images("./final_focused",clean_group_focused)

#import final datsets to be used for project
list_final_angry=import_images("./final_angry")
list_final_happy=import_images("./final_happy")
list_final_neutral=import_images("./final_neutral")
#list_final_focused=import_images("./final_focused") don't have dataset yet

#lists of lables for each emotion
list_angry_labels=label_images(list_final_angry,angry_label)
list_happy_labels=label_images(list_final_happy,happy_label)
list_neutral_labels=label_images(list_final_neutral,neutral_label)
# list_focused_labels=label_images(list_final_focused,focused_label)


#convert images in lists to numpy array, they're still stored in a list
angry_images_as_array = convert_images_to_numpy_array(list_final_angry) #these are lists that contain arrays
happy_images_as_array = convert_images_to_numpy_array(list_final_happy)
neutral_images_as_array = convert_images_to_numpy_array(list_final_neutral)
# focused_images_as_array = convert_images_to_numpy_array(list_final_focused) don't have dataset yet

#add all lists together to make one big list, need to add focused_images_as_array
final_list_of_all_images = angry_images_as_array + happy_images_as_array + neutral_images_as_array#+focused_images_as_array
final_list_of_all_labels = list_angry_labels + list_happy_labels + list_neutral_labels#+list_focused_labels

#************************************
#************************************
#Caspar and Jonny, I think you guys are going to need the numpy arays that I made below this comment
#in your data_visualization.py file, I'm not done yet though I still have to find a focused dataset
#and add pictures of us to the dataset

#these are numpy arrays of each emotion that store each image as a numpy array which stores the pixel values
array_of_angry = np.array(angry_images_as_array)
array_of_happy = np.array(happy_images_as_array)
array_of_neutral = np.array(neutral_images_as_array)
# array_of_focused = np.array(focused_images_as_array)

#numpyarrays storing labels for each emotion
array_angry_labels=np.array(list_angry_labels)
array_happy_labels=np.array(list_happy_labels)
array_neutral_labels=np.array(list_neutral_labels)
#array_focused_labels = np.array(list_focused_labels)

#numpy array containing whole dataset together and all the labels together
final_array_of_all_images = np.array(final_list_of_all_images)
final_array_of_all_labels = np.array(final_list_of_all_labels)


