import os
from PIL import Image
from matplotlib import pyplot as plt

emotions = ["angry", "focused", "happy", "neutral"]

# Define the source folder and the new folder paths

gender_options = ['m', 'f', 'o']
age_options = ['y', 'm', 's']

gender = ""
age = ""

for emotion in emotions:
    source_folder = f"./Final_clean_dataset/final_{emotion}"
    # Iterate through each image in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") :
            image_path = os.path.join(source_folder, filename)

            # Load and display the image
            img = Image.open(image_path)

            # Display the image using matplotlib
            plt.imshow(img)
            plt.axis('off')  # Hide axes
            plt.show()

            print(f"filename: {filename}")

            # Prompt the user to enter gender and age for the current image
            gender = input("Enter gender for this image (m, f, o): ").lower()
            while gender not in gender_options:
                gender = input("Invalid input. Enter gender for this image (m, f, o): ").lower()

            age = input("Enter age group for this image (y, m, s): ").lower()
            while age not in age_options:
                age = input("Invalid input. Enter age group for this image (y, m, s): ").lower()

            # Define the destination folders based on user input
            dest_gender_folder = f'./p3_{gender}/{emotion}_{gender}'
            dest_age_folder = f'./p3_{age}/{emotion}_{age}'

            # Create destination folders if they don't exist
            os.makedirs(dest_gender_folder, exist_ok=True)
            os.makedirs(dest_age_folder, exist_ok=True)

            # Check if the image already exists in the destination folders
            dest_gender_path = os.path.join(dest_gender_folder, filename)
            dest_age_path = os.path.join(dest_age_folder, filename)

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

            # Close the image display
            img.close()

    print("Images have been moved to the new folders.")
