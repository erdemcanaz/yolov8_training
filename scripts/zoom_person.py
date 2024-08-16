import os, copy, cv2
from pathlib import Path
from ultralytics import YOLO
import numpy as np

# Paths to the folders
EXPORT_FOLDER_PATH = Path(__file__).resolve().parent.parent / "exports"
ZOOM_FILTER_SHAPE= (320, 320)
FINAL_ZOOMED_FRAME_SHAPE = (640, 640)

SHOW_ZOOMED_PERSON = False
WAIT_SHOW_ZOOMED_PERSON = 250
POSE_DETECTOR_CONFIDENCE_THRESHOLD = 0.5

read_folder = input("Enter the path to the folder containing images: ")
#================================================================================================


class YOLOWrapper:
    def __init__(self, model_path:str=None) -> None:
        self.yolo_object = YOLO(model_path)
        self.recent_detection_results:dict = None # This will be a list of dictionaries, each dictionary will contain the prediction results for a single detection

    def __clear_recent_detection_results(self):
        self.recent_detection_results = {          
            "normalized_bboxes": [] # List of normalized bounding boxes in the format [x1n, y1n, x2n, y2n, bbox_confidence, class_name]
        }

    def detect_frame(self, frame:np.ndarray = None, confidence_threshold:float = POSE_DETECTOR_CONFIDENCE_THRESHOLD) -> None:
        self.__clear_recent_detection_results()

        detections = self.yolo_object(frame, task = "detect",)[0]
        for detection in detections:
            boxes = detection.boxes
            box_cls_no = int(boxes.cls.cpu().numpy()[0])
            box_cls_name = self.yolo_object.names[box_cls_no]            
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxyn = boxes.xyxyn.cpu().numpy()[0]

            if box_conf < confidence_threshold: continue
            
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
    
def zoom_person(image:np.ndarray, bbox:tuple, zoom_filter_shape:tuple=ZOOM_FILTER_SHAPE, final_zoomed_frame_shape:tuple=FINAL_ZOOMED_FRAME_SHAPE) -> np.ndarray:
    x1n, y1n, x2n, y2n = bbox
    x1, y1, x2, y2 = int(x1n*image.shape[1]), int(y1n*image.shape[0]), int(x2n*image.shape[1]), int(y2n*image.shape[0])

    zoom_width = min(zoom_filter_shape[0], image.shape[1])
    zoom_height = min(zoom_filter_shape[1], image.shape[0])
    zoom_filter_shape = (zoom_width, zoom_height)

    #check if person fits in the zoom area, if not person is already big enough, no need to zoom. format to res
    if x2 - x1 > zoom_width or y2 - y1 > zoom_height:
        image = cv2.resize(image, final_zoomed_frame_shape) 
        return image
    
    bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
    zoom_area_bbox = (bbox_center[0] - zoom_filter_shape[0]//2, bbox_center[1] - zoom_filter_shape[1]//2, bbox_center[0] + zoom_filter_shape[0]//2, bbox_center[1] + zoom_filter_shape[1]//2)

    if zoom_area_bbox[0] < 0: #shift the filter to the right
        shift_x = abs(zoom_area_bbox[0])
        zoom_area_bbox = (0, zoom_area_bbox[1], zoom_area_bbox[2] + shift_x, zoom_area_bbox[3])
    if zoom_area_bbox[2] > image.shape[1]: #shift the filter to the left
        shift_x = zoom_area_bbox[2] - image.shape[1]
        zoom_area_bbox = (zoom_area_bbox[0] - shift_x, zoom_area_bbox[1], image.shape[1], zoom_area_bbox[3])
    if zoom_area_bbox[1] < 0: #shift the filter down
        shift_y = abs(zoom_area_bbox[1])
        zoom_area_bbox = (zoom_area_bbox[0], 0, zoom_area_bbox[2], zoom_area_bbox[3] + shift_y)
    if zoom_area_bbox[3] > image.shape[0]: #shift the filter up
        shift_y = zoom_area_bbox[3] - image.shape[0]
        zoom_area_bbox = (zoom_area_bbox[0], zoom_area_bbox[1] - shift_y, zoom_area_bbox[2], image.shape[0])

    zoomed_roi = image[zoom_area_bbox[1]:zoom_area_bbox[3], zoom_area_bbox[0]:zoom_area_bbox[2]]
    zoomed_roi = cv2.resize(zoomed_roi, final_zoomed_frame_shape)
    return copy.deepcopy(zoomed_roi)

delete_files_in_folder(EXPORT_FOLDER_PATH)
pose_model = YOLOWrapper(Path(__file__).resolve().parent.parent / "trained_yolo_models" / "yolov8n-pose.pt")
image_paths = return_image_paths(read_folder)

for image_path in image_paths:
    image = load_image(image_path)

    pose_model.detect_frame(frame = image, confidence_threshold = POSE_DETECTOR_CONFIDENCE_THRESHOLD)    
    for person_no, person_bbox in enumerate(pose_model.get_recent_detection_results()["normalized_bboxes"]):
        zoomed_resized_frame = zoom_person(image, person_bbox[:4])
        
        # Create the new filename with the prefix
        original_filename = Path(image_path).name
        new_filename = f"zoomed_{person_no}_{original_filename}"

        # Save the image with the new filename
        cv2.imwrite(str(EXPORT_FOLDER_PATH / new_filename), zoomed_resized_frame)

        if SHOW_ZOOMED_PERSON:
            cv2.imshow("Zoomed Person", zoomed_resized_frame)
            key = cv2.waitKey(WAIT_SHOW_ZOOMED_PERSON) & 0xFF
            if key == 27:  # ESC key to break the loop
                break

  
cv2.destroyAllWindows()