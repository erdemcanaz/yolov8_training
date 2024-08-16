import os
import cv2, torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np

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
    
class YOLOWrapper:
    def __init__(self, model_path:str=None) -> None:
        self.yolo_object = YOLO(model_path)
        self.recent_detection_results:dict = None # This will be a list of dictionaries, each dictionary will contain the prediction results for a single detection

    def __clear_recent_detection_results(self):
        self.recent_detection_results = {          
            "normalized_bboxes": [] # List of normalized bounding boxes in the format [x1n, y1n, x2n, y2n, bbox_confidence, class_name]
        }

    def detect_frame(self, frame:np.ndarray = None):
        self.__clear_recent_detection_results()

        detections = self.yolo_object(frame, task = "detect",)[0]
        for detection in detections:
            boxes = detection.boxes
            box_cls_no = int(boxes.cls.cpu().numpy()[0])
            box_cls_name = self.yolo_object.names[box_cls_no]            
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxyn = boxes.xyxyn.cpu().numpy()[0]
            self.recent_detection_results["normalized_bboxes"].append([box_xyxyn[0], box_xyxyn[1], box_xyxyn[2], box_xyxyn[3], box_conf, box_cls_name])
    
    def get_recent_detection_results(self) -> dict:
        return self.recent_detection_results
    
    def draw_recent_bbox_on_frame(self, frame:np.ndarray, color:tuple=(0, 255, 0), thickness:int=2, text_thickness:int = 3) -> np.ndarray:
        for bbox in self.recent_detection_results["normalized_bboxes"]:
            x1, y1, x2, y2 = bbox[0]*frame.shape[1], bbox[1]*frame.shape[0], bbox[2]*frame.shape[1], bbox[3]*frame.shape[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.putText(frame, f"{bbox[4]:.2f} {bbox[5]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
      
    def is_any_detection(self) -> bool:
        return len(self.recent_detection_results["normalized_bboxes"]) > 0
    
IMAGE_FOLDER_PATH = input("Enter the path to the folder containing images: ")
MODEL_PATH = input("Enter the path to the YOLO model: ")

image_paths = return_image_paths(IMAGE_FOLDER_PATH)
yolo_model_to_test = YOLOWrapper(MODEL_PATH)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
for image_path in image_paths:
    image = load_image(image_path)

    yolo_model_to_test.detect_frame(frame = image)
    yolo_model_to_test.draw_recent_bbox_on_frame(frame = image)
    
    resized_image = cv2.resize(image, (800, 600))
    cv2.imshow("Image", resized_image)

    wait_time = 2500 if yolo_model_to_test.is_any_detection() else 50

    key = cv2.waitKey(wait_time) & 0xFF
    if key == 27:  # 27 is the ASCII code for the ESC key
        break

cv2.destroyAllWindows()



