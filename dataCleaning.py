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
    for filename in os.listdir(folder_path):    #used chatgpt, I have to cite it
        if filename.endswith('.jpg'):
            with Image.open(os.path.join(folder_path, filename)) as img:
                original_images.append(img.copy())

    return original_images

def clean_images(list_of_dirty_pictures):
    clean_pictures=[]
    for img in list_of_dirty_pictures:
        resized_image= img.resize(new_size)
        enhancer = ImageEnhance.Brightness(resized_image)
        brighter_image = enhancer.enhance(new_brightness)
        clean_pictures.append(brighter_image)

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

def convert_images_to_numpy(list_of_images_to_convert):
    list_of_numpy_images=[]
    for img in list_of_images_to_convert:
        x = np.array(img)
        list_of_numpy_images.append(x)

    return list_of_numpy_images

#import, process and then export happy images
original_happy_images=import_images("./happy")
clean_happy_images=clean_images(original_happy_images)
export_images("./new_happy",clean_happy_images)
clean_happy_labels=label_images(clean_happy_images,happy_label)

#import, process and then export angry images
original_angry_images=import_images("./angry")
clean_angry_images=clean_images(original_angry_images)
export_images("./new_angry",clean_angry_images)
clean_angry_labels=label_images(clean_angry_images,angry_label)


#import, process and then export neutral images
original_neutral_images=import_images("./neutral")
clean_neutral_images=clean_images(original_neutral_images)
export_images("./new_neutral",clean_neutral_images)
clean_neutral_labels=label_images(clean_neutral_images,neutral_label)


#************************************
#************************************
#Caspar and Jonny, I think you guys are going to need the numpy arays that I made below this comment
#in your data_visualization.py file, I'm not done yet though I still have to find a focused dataset
#and add pictures of us to the dataset



#convert lists containing images and labels into numpy arrays
array_of_clean_happy_images=np.array(clean_happy_images)
array_of_clean_happy_labels=np.array(clean_happy_labels)

#convert lists containing images and labels into numpy arrays
array_of_clean_angry_images=np.array(clean_angry_images)
array_of_clean_angry_labels=np.array(clean_angry_labels)

#convert lists containing images and labels into numpy arrays
array_of_clean_neutral_images=np.array(clean_neutral_images)
array_of_clean_neutral_labels=np.array(clean_neutral_labels)