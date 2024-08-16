import os
from PIL import Image
import uuid, datetime

def rename_images(folder_path, title:str= None):
    # List all files in the directory
    if title is None or title == "":
        title = "no_title"

    files = os.listdir(folder_path)
    
    # Filter only files with image extensions (you can add more extensions if needed)
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    #images = [file for file in files if file.lower().endswith(image_extensions)]
    images = [file for file in files]
    
    # Rename each image
    for index, image in enumerate(images, start=1):
        # Get the file extension
        file_extension = os.path.splitext(image)[1]
        
        # Create the new name with leading zeros up to 5 digits
        new_name = f"{title}_{index:05}-{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M')}_{uuid.uuid4()}{file_extension}"
        
        # Get the full old and new file paths
        old_file = os.path.join(folder_path, image)
        new_file = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} to {new_file}")

def convert_to_jpg(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert image to RGB (necessary for some formats)
        img = img.convert('RGB')
        
        # Change the file extension to .jpg
        new_image_path = os.path.splitext(image_path)[0] + '.jpg'
        
        # Save the image in JPG format
        img.save(new_image_path, 'JPEG')
        print(f"Converted: {new_image_path}")
        
        # Delete the old file
        os.remove(image_path)
        print(f"Deleted: {image_path}")

# Example usage
folder_path = input("Resimlerin konumu: ")
title = input("Resim ön adı: ")

rename_images(folder_path, title)

# Convert images to JPG
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if not file.lower().endswith('.jpg'):
        convert_to_jpg(file_path)