import os
import cv2
# Set your directories here
images_dir = input("Enter the path to the images folder: ")
labels_dir = input("Enter the path to the labels folder: ")

# Get list of all files in images and labels folders
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

# Convert to sets of filenames without extensions
image_basenames = set(os.path.splitext(f)[0] for f in image_files)
label_basenames = set(os.path.splitext(f)[0] for f in label_files)

# Find images without labels and labels without images
unpaired_images = image_basenames - label_basenames
unpaired_labels = label_basenames - image_basenames - {'classes'}


number_of_unpaired_images = len(unpaired_images)
number_of_unpaired_labels = len(unpaired_labels)

print(f"Number of unpaired images: {number_of_unpaired_images}")
print(f"Number of unpaired labels: {number_of_unpaired_labels}")
should_continue = input("Do you want to continue? (y/n): ")
if should_continue != 'y':
    print("Exiting...")
    exit()
    
# Show and delete unpaired images if 'x' is pressed
for image in unpaired_images:
    image_path = os.path.join(images_dir, image + '.jpg')
    if not os.path.exists(image_path):  # If the image is .png instead of .jpg
        image_path = os.path.join(images_dir, image + '.png')
    
    # Check if the image exists
    if os.path.exists(image_path):
        # Load the image using OpenCV
        img = cv2.imread(image_path)
        
        # Display the image
        cv2.imshow('Image', img)
        
        # Wait for a key press
        key = cv2.waitKey(0) & 0xFF
        
        # If 'x' is pressed, delete the image
        if key == ord('x'):
            print(f"Deleting unpaired image: {image_path}")
            os.remove(image_path)
        
        # Close the image window
        cv2.destroyAllWindows()

# Delete unpaired labels
for label in unpaired_labels:
    if label == 'classes':
        continue
    label_path = os.path.join(labels_dir, label + '.txt')
    print(f"Deleting unpaired label: {label_path}")
    os.remove(label_path)

print("Validation complete. Unpaired files have been deleted.")