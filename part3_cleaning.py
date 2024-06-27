from PIL import Image, ImageEnhance
import os


new_size=(244,244)
new_brightness=1.2
image_counter=0
new_name="dataset_augmentation_image"

folder_names = ["./senior_angry","./senior_focused","./senior_happy","./senior_neutral",
                "./other_angry","./other_focused","./other_happy","./other_neutral"]

def rename_images(folder_path):
    global image_counter
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                new_filename = f"{new_name}_{image_counter}.jpg"

                old_file_path = os.path.join(root, filename)
                new_file_path = os.path.join(root, new_filename)

                os.rename(old_file_path,new_file_path)
                image_counter += 1


def import_images(folder_path):
    images = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(root, filename)
                img = Image.open(file_path)
                images.append(img)

    return images

def clean_images(list_of_dirty_pictures):
    clean_pictures=[]
    for img in list_of_dirty_pictures:
        resized_image = img.resize(new_size)
        enhancer = ImageEnhance.Brightness(resized_image)   #chatGPT was used for this line
        brighter_image = enhancer.enhance(new_brightness)   #chatGPT was used for this line
        greyscale_image = brighter_image.convert("L")       #chatGPT was used for this line
        clean_pictures.append(greyscale_image)

    return clean_pictures

def export_images(new_folder_path, list_of_images_to_export, rgb):
    os.makedirs(new_folder_path, exist_ok=True) #chatGPT was used for this line
    global image_counter
    for img in list_of_images_to_export:
        if rgb:
            img = img.convert("RGB")  # ChatGPT was used for this line
        output_path = os.path.join(new_folder_path, f"{new_name}_{image_counter}.jpg")  #chatGPT was used for this line
        img.save(output_path)
        image_counter+=1

    return


senior_angry = import_images("./senior_angry")
clean_senior_angry=clean_images(senior_angry)
export_images("./part3_senior/angry_s",clean_senior_angry,False)

senior_focused = import_images("./senior_focused")
clean_senior_focused = clean_images(senior_focused)
export_images("./part3_senior/focused_s",clean_senior_focused,False)

senior_happy = import_images("./senior_happy")
clean_senior_happy = clean_images(senior_happy)
export_images("./part3_senior/happy_s",clean_senior_happy,False)

senior_neutral = import_images("./senior_neutral")
clean_senior_neutral = clean_images(senior_neutral)
export_images("./part3_senior/neutral_s",clean_senior_neutral,False)

other_angry = import_images("./other_angry")
clean_other_angry = clean_images(other_angry)
export_images("./part3_other/angry_o",clean_other_angry,False)

other_focused = import_images("./other_focused")
clean_other_focused = clean_images(other_focused)
export_images("./part3_other/focused_o",clean_other_focused,False)

other_happy = import_images("./other_happy")
clean_other_happy = clean_images(other_happy)
export_images("./part3_other/happy_o",clean_other_happy,False)

other_neutral = import_images("./other_neutral")
clean_other_neutral = clean_images(other_neutral)
export_images("./part3_other/neutral_o",clean_other_neutral,False)



