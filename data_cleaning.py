from PIL import Image,ImageEnhance
import os
import numpy as np
import random
import shutil

new_size=(244,244)
new_brightness=1.2
angry_label=1
neutral_label=2
focused_label=3
happy_label=4
image_counter=0

def select_random_images(src_folder, dst_folder, num_images):
    os.makedirs(dst_folder, exist_ok=True)
    all_files = os.listdir(src_folder)
    selected_files = random.sample(all_files, num_images)
    # Copy the selected files to the destination folder
    for file_name in selected_files:
        src_file_path = os.path.join(src_folder, file_name)
        dst_file_path = os.path.join(dst_folder, file_name)
        shutil.copy(src_file_path, dst_file_path)
    return


def import_images(folder_path):
    original_images = []
    import_counter=0
    for filename in os.listdir(folder_path):
        if (import_counter==510):
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
        greyscale_image = brighter_image.convert("L")
        clean_pictures.append(greyscale_image)

    return clean_pictures


def export_images(new_folder_path, list_of_images_to_export):
    os.makedirs(new_folder_path, exist_ok=True)
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
#choose 500 random images and save them
select_random_images("./focused","./random_focused",500)
select_random_images("./angry","./random_angry",500)
select_random_images("./happy","./random_happy",500)
select_random_images("./neutral","./random_neutral",500)



# #***********************************
# #***********************************
# #import from fer datasets, focused dataset, and import group pictures
fer_happy=import_images("./random_happy")
fer_angry=import_images("./random_angry")
fer_neutral=import_images("./random_neutral")
focused = import_images("./random_focused")
group_happy=import_images("./group_happy")
group_angry=import_images("./group_angry")
group_neutral=import_images("./group_neutral")
group_engaged=import_images("./group_engaged")

# #make list of clean images for each class
clean_happy=clean_images(fer_happy)
clean_angry=clean_images(fer_angry)
clean_neutral=clean_images(fer_neutral)
clean_focused=clean_images(focused)
clean_group_happy=clean_images(group_happy)
clean_group_angry=clean_images(group_angry)
clean_group_neutral=clean_images(group_neutral)
clean_group_focused=clean_images(group_engaged)


# #Add group pics to appropriate datasets
clean_group_happy += clean_happy
clean_group_angry += clean_angry
clean_group_neutral += clean_neutral
clean_group_focused += clean_focused

# #save the images in the list to save dataset for project
export_images("./Final_dataset/final_happy",clean_group_happy)
export_images("./Final_dataset/final_angry",clean_group_angry)
export_images("./Final_dataset/final_neutral",clean_group_neutral)
export_images("./Final_dataset/final_focused",clean_group_focused)



