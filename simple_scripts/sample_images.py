import os
import random
import shutil

def get_image_files(folder):
    """Get the list of image files in the given folder."""
    return {f for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.png')}

def main():
    # Set the source and destination folders
    source_folder = input("Enter the source folder path: ")
    destination_folder = input("Enter the destination folder path: ")
    compare_folder = input("Enter the folder to compare: ")
    num_frames_to_copy = int(input("Enter the number of frames to copy: "))

    # Get the list of image files in the source and destination folders
    source_images = get_image_files(source_folder)
    destination_images = get_image_files(destination_folder)
    compare_folder_images = get_image_files(compare_folder)

    # Determine which images are not present in the destination folder
    images_to_copy = list(source_images - compare_folder_images)

    # If there are fewer images to copy than requested, adjust the number
    num_frames_to_copy = min(num_frames_to_copy, len(images_to_copy))

    # Randomly select the images to copy
    selected_images = random.sample(images_to_copy, num_frames_to_copy)

    # Copy the selected images
    for image_file in selected_images:
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        shutil.copyfile(source_path, destination_path)
        print(f"Image file {image_file} copied to {destination_folder}")

if __name__ == "__main__":
    main()