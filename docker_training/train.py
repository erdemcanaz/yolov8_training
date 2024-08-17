from ultralytics import YOLO
import torch
from pathlib import Path
import os

def ensure_if_volume_is_proper(folder_path:str = None):
    if not os.path.exists(folder_path):
        raise Exception(f"Volume does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise Exception(f"Path is not a folder: {folder_path}")
    
def ensure_if_training_folder_is_proper(folder_path:str = None):
    if not os.path.exists(folder_path):
        raise Exception(f"Folder does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise Exception(f"Path is not a folder: {folder_path}")
    
def ensure_if_data_folder_to_be_trained_on_is_proper(folder_path:str = None):
    if not os.path.exists(folder_path + "/images"):
        raise Exception(f"Folder does not contain 'images' folder: {folder_path}")
    if not os.path.exists(folder_path + "/labels"):
        raise Exception(f"Folder does not contain 'labels' folder: {folder_path}")
    if not os.path.exists(folder_path + "/data.yaml"):
        raise Exception(f"Folder does not contain 'data.yaml' file: {folder_path}")
    
def get_available_models_in_folder(folder_path:str = None):
    available_models = []
    for file in os.listdir(folder_path):
        if file.endswith(".pt"):
            available_models.append(file)
    return available_models

def createa_training_results_folder_if_not_exists(folder_path:str = None):
    if not "training_results" in os.listdir(folder_path):
        os.mkdir(folder_path + "/training_results")
        
def start_training(data_folder:str = None , available_model_paths:list = None):
    print(f" torch.cuda.device_count(): {torch.cuda.device_count()}")
    torch.cuda.set_device(0) # Set to your desired GPU number

    # Initialize the model
    yolo_yamls= ["yolov8n.yaml", "yolov8s.yaml", "yolov8m.yaml", "yolov8l.yaml", "yolov8x.yaml"]
    for i, model in enumerate(yolo_yamls):
        print(f"{i}: {model}")
    yaml_index = int(input("Enter the index of the model to train: "))
    
    is_existing_model = True if input("Do you have an existing model to train on? (y/n): ")=="y" else False    
    if is_existing_model:
        #==== Option 1: Build from YAML and transfer pretrained weights
        for i, model_path in enumerate(available_model_paths):
            print(f"{i}: {model_path}")
        model_path_to_train_on = available_model_paths[int(input("Enter the index of the model to train on: "))]
        model = YOLO(yolo_yamls[yaml_index]).load(model_path_to_train_on)
    else:
        #==== Option 2: Train directly from the model definition
        model = YOLO(yolo_yamls[yaml_index])

    if torch.cuda.is_available():
        model.to('cuda')
        print("GPU (CUDA) is detected. Training will be done on GPU.")
    else:
        r = input("GPU (CUDA) is not detected or prefered. Should continue with CPU? (y/n):")
        if r != 'y':
            print("Exiting...")
            exit()

    experiment = input("Enter the name of your experiment: ")
    save_dir = data_folder / "training_results"
    data_yaml_file = data_folder / "data.yaml"


    class_indexes_to_train_on = input("Enter the classes to train on (comma separated): ").split(",")
    class_indexes_to_train_on = [int(class_index) for class_index in class_indexes_to_train_on if class_index.isdigit()]
    for i, class_index in enumerate(class_indexes_to_train_on):
        print(f"{i}: {class_index}")
    
    number_of_epochs = int(input("Enter the number of epochs: "))
    save_period = int(input("Enter the save period: "))
    batch_size = min(0.75, float(input("Enter the batch size (percentage): ")))

    model.train(
        data=data_yaml_file,
        classes = class_indexes_to_train_on,
        epochs=number_of_epochs, 
        save_dir=save_dir, 
        project=save_dir,
        name=experiment,
        imgsz=640,
        save_period = save_period,
        batch = batch_size, 
        plots = True,
        amp = True
    )

# Get the volume name
volume_name = input("Enter the name of the volume where 'training_data_folder' folder is: ")
volume_path = Path(__file__).resolve().parent.parent.parent / volume_name
ensure_if_volume_is_proper(volume_path)

# Get the training data folder
training_folder = volume_path / "training_data_folder"
ensure_if_training_folder_is_proper(folder_path = training_folder)

# Get the data folder
data_folder_name = input("Enter the path to the folder containing data: ")
data_folder = Path(__file__).resolve().parent.parent.parent / volume_name / data_folder_name
ensure_if_data_folder_to_be_trained_on_is_proper(folder_path = data_folder)
createa_training_results_folder_if_not_exists(folder_path=  data_folder)
available_model_paths = get_available_models_in_folder(folder_path = training_folder) # Path of every .pt file in the folder

# Start training
start_training(data_folder = data_folder ,available_model_paths = available_model_paths)


