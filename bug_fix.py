import os
from PIL import Image
from matplotlib import pyplot as plt


#This script is to help label our images by showing them and asking us to assign them to a gender class and age class
#there are three classes for gender: m=male, f=female, o=other
#there are three classes for age: y=young, m=middle aged, s=senior


emotion = "neutral"

gender_options = ['m', 'f', 'o', 'q']
age_options = ['y', 'z', 's']

gender = ""
age = ""

middleaged = "middleaged"

source_folder = f"./part3_male/{emotion}_m"

print("###################################################")
print("###################################################")
print("Once you start labelling an emotion don't close this program until all the images for that emotion have been labelled")
print("if you stop running the program once you start labelling an emotions images you'll have to start over")
print("Also once the image is displayed on your computer you must close the image before entering the classes")
print("otherwise it won't work")
print("###################################################")
print("###################################################")
print("###################################################")
print("###################################################")

#this for loop goes through each image in the emotion folder in the Final_clean_dataset
for filename in os.listdir(source_folder):
    if filename.endswith(".jpg") :
        image_path = os.path.join(source_folder, filename)
        print(image_path)
        img = Image.open(image_path)

        if os.path.exists(f"./p3_middleaged/{emotion}_middleaged/{filename}"):
            print(f"Image {filename} already in middle aged folder.")
            continue

        #This will display the image automatically on your computer
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        #this is where you classify the gender of the image that popped up and the age
        gender = input("Enter gender for this image (m, f, o, q): ").lower()

        while gender not in gender_options:
            gender = input("Invalid input. Enter gender for this image (m, f, o, q): ").lower()

        if gender == "q":
            continue

        age = input("Enter age group for this image (y, z, s): ").lower()
        while age not in age_options:
            age = input("Invalid input. Enter age group for this image (y, z, s): ").lower()

        if age == "z":
            dest_age_folder = f'./p3_{middleaged}/{emotion}_{middleaged}'

            os.makedirs(dest_age_folder, exist_ok=True)

            dest_age_path = os.path.join(dest_age_folder, filename)

            if not os.path.exists(dest_age_path):
                img.save(dest_age_path)
                print(f"Image {filename} moved to {dest_age_folder}.")
            else:
                print(f"Image {filename} already exists in {dest_age_folder}.")

        if gender == "f":
            os.remove(image_path)

        img.close()


print(f"{emotion} Images have been moved to the new folders. Change the emotion on line 11 to the next emotion")
print("for example, if you want to label all the images in the final_focused folder, make line 11 look like:  ")
print('emotion = "focused"')
print("ITS IMPORTANT TO GO THROUGH THE FOLDERS IN THE SAME ORDER AS THE STRUCTURE IN OUR PROJECT, IF YOU DON'T IT WILL MESS UP OUR PREDICTIONS")
print("SO DO ANGRY, THEN FOCUSED, THEN HAPPY AND THEN NEUTRAL")
