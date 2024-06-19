import os
from PIL import Image
from matplotlib import pyplot as plt


#This script is to help label our images by showing them and asking us to assign them to a gender class and age class
#there are three classes for gender: m=male, f=female, o=other
#there are three classes for age: y=young, m=middle aged, s=senior


emotion = "angry"

gender_options = ['m', 'f', 'o']
age_options = ['y', 'm', 's']

gender = ""
age = ""

source_folder = f"./Final_clean_dataset/final_{emotion}"

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

        img = Image.open(image_path)

        #This will display the image automatically on your computer
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        print(f"filename: {filename}")

        #this is where you classify the gender of the image that popped up and the age
        gender = input("Enter gender for this image (m, f, o): ").lower()
        while gender not in gender_options:
            gender = input("Invalid input. Enter gender for this image (m, f, o): ").lower()

        age = input("Enter age group for this image (y, m, s): ").lower()
        while age not in age_options:
            age = input("Invalid input. Enter age group for this image (y, m, s): ").lower()

        #this chooses the path to save the image to based on which class it is
        dest_gender_folder = f'./p3_{gender}/{emotion}_{gender}'
        dest_age_folder = f'./p3_{age}/{emotion}_{age}'

        os.makedirs(dest_gender_folder, exist_ok=True)
        os.makedirs(dest_age_folder, exist_ok=True)

        dest_gender_path = os.path.join(dest_gender_folder, filename)
        dest_age_path = os.path.join(dest_age_folder, filename)

        #this makes sure that the image hasn't already been classified
        if not os.path.exists(dest_gender_path):
            img.save(dest_gender_path)
            print(f"Image {filename} moved to {dest_gender_folder}.")
        else:
            print(f"Image {filename} already exists in {dest_gender_folder}.")

        if not os.path.exists(dest_age_path):
            img.save(dest_age_path)
            print(f"Image {filename} moved to {dest_age_folder}.")
        else:
            print(f"Image {filename} already exists in {dest_age_folder}.")

        img.close()


print(f"{emotion} Images have been moved to the new folders. Change the emotion on line 11 to the next emotion")
print("for example, if you want to label all the images in the final_focused folder, make line 11 look like:  ")
print('emotion = "focused"')
print("ITS IMPORTANT TO GO THROUGH THE FOLDERS IN THE SAME ORDER AS THE STRUCTURE IN OUR PROJECT, IF YOU DON'T IT WILL MESS UP OUR PREDICTIONS")
print("SO DO ANGRY, THEN FOCUSED, THEN HAPPY AND THEN NEUTRAL")
