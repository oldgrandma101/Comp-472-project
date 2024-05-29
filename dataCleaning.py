from PIL import Image,ImageEnhance
import os

new_size=(244,244)
new_brightness=1.2
def import_images(folder_path):
    original_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            with Image.open(os.path.join(folder_path, filename)) as img:
                original_images.append(img.copy())


    return original_images

def clean_images(new_size, new_brightness, list_of_dirty_pictures):
    clean_pictures=[]
    for img in list_of_dirty_pictures:
        resized_image= img.resize(new_size)
        enhancer = ImageEnhance.Brightness(resized_image)
        brighter_image = enhancer.enhance(new_brightness)
        clean_pictures.append(brighter_image)

    return clean_pictures


def export_images(new_folder_path, list_of_images_to_export):
    os.makedirs(new_folder_path, exist_ok=True)
    counter=0
    for img in list_of_images_to_export:
        output_path = os.path.join(new_folder_path, f"image_{counter}.jpg")
        img.save(output_path)
        counter+=1

    return


original_happy_images = import_images("./happy")
clean_happy_images = clean_images(new_size, new_brightness, original_happy_images)
export_images("./new_happy", clean_happy_images)