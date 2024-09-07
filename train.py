from ultralytics import YOLO
import torch

def main():
    print(f" torch.cuda.device_count(): {torch.cuda.device_count()}")
    torch.cuda.set_device(0) # Set to your desired GPU number

    is_existing_model = True if input("Do you have an existing model to train on? (y/n): ")=="y" else False
    if is_existing_model:
        #==== Option 1: Build from YAML and transfer pretrained weights
        model_path_to_train_on = input("Enter the path to the model to train on ( original one is not effected ) : ")
        model = YOLO('yolov8n.yaml').load(model_path_to_train_on)
    else:
        #==== Option 2: Train directly from the model definition
        model = YOLO('yolov8n.yaml')

    RUN_ON_CUDA = True
    if RUN_ON_CUDA and torch.cuda.is_available():
        model.to('cuda')
        print("GPU (CUDA) is detected. Training will be done on GPU.")
    else:
        r = input("GPU (CUDA) is not detected or prefered. Should continue with CPU? (y/n):")
        if r != 'y':
            print("Exiting...")
            exit()

    #Train the model
    experiment = input("Enter the name of your experiment: ")
    save_dir = input("Enter the path to your save directory: ")
    yaml_file = input("Enter the path to your yaml file: ")

    model.train(
        data=yaml_file,
        classes = [0,1],
        epochs=150, 
        save_dir=save_dir, 
        project=save_dir,
        name=experiment,
        imgsz=640,
        save_period = 100,
        batch = 0.7, 
        plots = True,
        amp = False, # Nan Reading if set to TRUE -> BUG: https://stackoverflow.com/questions/75178762/i-got-nan-for-all-losses-while-training-yolov8-model


        #Augmentation (https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters)
        hsv_h=0.0, #(0.0 - 1.0) Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.
        hsv_s=0.0, #(0.0 - 1.0) Adjusts the saturation of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.
        hsv_v=0.0, #(0.0 - 1.0) Adjusts the value of the image by a fraction of the color wheel, introducing color variability. Helps the model generalize across different lighting conditions.
        degrees=180, #(0.0 - 180.0) Rotates the image by a random angle within the specified range. Helps the model generalize across different orientations.
        translate = 0.0, #(0.0 - 1.0) Translates the image by a random fraction of the image size. Helps the model generalize across different positions.
        scale = 0.0, #(0.0 - 1.0) Scales the image by a random factor. Helps the model generalize across different scales.
        shear = 0.0, #(0.0 - 1.0) Shears the image by a random angle within the specified range. Helps the model generalize across different perspectives.
        perspective = 0.0, #(0.0 - 1.0) Distorts the image by a random fraction of the image size. Helps the model generalize across different perspectives.
        flipud = 0.0, #(0.0 - 1.0) Flips the image vertically. Helps the model generalize across different orientations.
        fliplr = 0.0, #(0.0 - 1.0) Flips the image horizontally. Helps the model generalize across different orientations.
        bgr = False, #Converts the image from RGB to BGR. May improve performance on some hardware.
        mosaic = 0, #(0.0 - 1.0) Adds mosaic augmentation to the image. Helps the model generalize across different positions, scales, and perspectives.
        mixup = 0, #(0.0 - 1.0) Adds mixup augmentation to the image. Helps the model generalize across different objects.
        copy_paste = 0.0, #(0.0 - 1.0) Adds copy-paste augmentation to the image. Helps the model generalize across different objects.
        erasing = 1, #(0.0 - 1.0) Adds random erasing augmentation to the image. Helps the model generalize across different objects.
        crop_fraction = 0.0, #(0.0 - 1.0) Crops the image by a random fraction of the image size. Helps the model generalize across different positions.
        
    )

if __name__ == '__main__':
    main()


