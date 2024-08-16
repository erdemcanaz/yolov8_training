import os
import cv2
from pathlib import Path

def delete_files_in_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through each file
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            # Check if it's a file and delete it
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

def return_image_paths(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    image_paths= []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image_paths.append(image_path)
    
    return image_paths

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        print(f"Loaded: {image_path}")
        return image
    else:
        raise Exception(f"Failed to load: {image_path}")    
    
# Paths to the folders
EXPORT_FOLDER_PATH = Path(__file__).resolve().parent.parent / "exports"
read_folder = input("Enter the path to the folder containing images: ")

# Delete all files in the specified folder
delete_files_in_folder(EXPORT_FOLDER_PATH)

# Read all images from the specified folder
image_paths = return_image_paths(read_folder)

# Optionally, process the images (example: display the first image)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

for image_path in image_paths:
    image = load_image(image_path)
    cv2.imshow("Image", image)
    cv2.waitKey(50)
    cv2.destroyAllWindows()