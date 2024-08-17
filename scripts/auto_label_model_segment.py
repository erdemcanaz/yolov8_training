import os, copy, cv2, pprint
from pathlib import Path
from ultralytics import YOLO, SAM
import numpy as np

# Paths to the folders
EXPORT_FOLDER_PATH = Path(__file__).resolve().parent.parent / "exports" # where labels will be saved

SHOW_ZOOMED_PERSON = True
WAIT_SHOW_ZOOMED_PERSON = 0
DETECTOR_CONFIDENCE_THRESHOLD = 0.5

read_folder = input("Enter the path to the folder containing images: ")
#================================================================================================


class YOLOWrapper:
    def __init__(self, model_path:str=None) -> None:
        self.yolo_object = YOLO(model_path)
        self.recent_detection_results:dict = None # This will be a list of dictionaries, each dictionary will contain the prediction results for a single detection

    def __clear_recent_detection_results(self):
        self.recent_detection_results = {          
            "normalized_bboxes": [], # List of normalized bounding boxes in the format [x1n, y1n, x2n, y2n, bbox_confidence, class_name]
            "keypoints": [] # List of keypoints in the format [x, y, confidence]
        }

    def detect_frame(self, frame:np.ndarray = None, confidence_threshold:float = DETECTOR_CONFIDENCE_THRESHOLD) -> None:
        self.__clear_recent_detection_results()

        detections = self.yolo_object(frame, task = "detect",)[0]
        for detection in detections:
            boxes = detection.boxes
            box_cls_no = int(boxes.cls.cpu().numpy()[0])
            box_cls_name = self.yolo_object.names[box_cls_no]            
            box_conf = boxes.conf.cpu().numpy()[0]
            box_xyxyn = boxes.xyxyn.cpu().numpy()[0]

            if box_conf < confidence_threshold: continue
            
            self.recent_detection_results["normalized_bboxes"].append([box_xyxyn[0], box_xyxyn[1], box_xyxyn[2], box_xyxyn[3], box_conf, box_cls_name, box_cls_no])
    
    def pose_frame(self, frame:np.ndarray = None, confidence_threshold:float = DETECTOR_CONFIDENCE_THRESHOLD) -> None:
        self.__clear_recent_detection_results()

        pose_results = self.yolo_object(frame, task = "pose")[0]
        for pose_result in pose_results:
            boxes = pose_result.boxes
            box_cls_no = int(boxes.cls.cpu().numpy()[0])
            box_cls_name = self.yolo_object.names[box_cls_no]            
            box_conf = boxes.conf.cpu().numpy()[0]
            if box_conf < confidence_threshold: continue
            box_xyxyn = boxes.xyxyn.cpu().numpy()[0]
            
            self.recent_detection_results["normalized_bboxes"].append([box_xyxyn[0], box_xyxyn[1], box_xyxyn[2], box_xyxyn[3], box_conf, box_cls_name])
        
            key_points = pose_result.keypoints  # Keypoints object for pose outputs
            keypoint_confs = key_points.conf.cpu().numpy()[0]
            keypoints_xyn = key_points.xyn.cpu().numpy()[0]   

            dlist = []
            for i in range(len(keypoint_confs)):
                keypoint_xn = keypoints_xyn[i][0]
                keypoint_yn = keypoints_xyn[i][1]
                keypoint_conf = keypoint_confs[i] if keypoint_xn > 0 or keypoint_yn > 0 else -keypoint_confs[i]

                dlist.append([keypoint_xn, keypoint_yn, keypoint_conf])
               
            self.recent_detection_results["keypoints"].append(dlist)

    def get_keypoint_index_by_name(self, keypoint_name:str) -> int:
        keypoints = ["nose", "right_eye", "left_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow" ,"right_elbow","left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        return keypoints.index(keypoint_name)
    
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

class SAMWrapper:
    def __init__(self, model_path:str=None) -> None:
        self.sam_object = SAM(model_path)
        self.recent_segmentation_results:dict = None
    
    def __clear_recent_segmentation_results(self):
        self.recent_segmentation_results = {
            "masks": [], # List of masks in the format [[x1n, y1n], [x2n, y2n], ...]
        }

    def get_mask_bbox(self, mask:np.ndarray) -> list:
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        return [x, y, x+w, y+h]
    
    def segment_by_rect(self, frame:np.ndarray = None, normalized_bbox_coordinates:list = None, class_name:str = None) -> None:
        self.__clear_recent_segmentation_results()
        
        bbox = [normalized_bbox_coordinates[0]*frame.shape[1], normalized_bbox_coordinates[1]*frame.shape[0], normalized_bbox_coordinates[2]*frame.shape[1], normalized_bbox_coordinates[3]*frame.shape[0]]
        segmentation_results = self.sam_object(frame, bboxes=[bbox], show=False)[0]
        
        masks_xy = segmentation_results.masks.xy
        max_x = float('-inf')
        max_y = float('-inf')
        min_x = float('inf')
        min_y = float('inf')

        for mask_xy in masks_xy:
            for x, y in mask_xy:                  
                max_x = max(max_x, x)
                max_y = max(max_y, y)
                min_x = min(min_x, x)
                min_y = min(min_y, y)

        color = (0,255,0) if class_name == "hard_hat" else (0,0,255)
        cv2.putText(frame, f"{class_name}", (int(min_x), int(min_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), color, 2)   
        return [int(min_x), int(min_y), int(max_x), int(max_y)]        
        
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

def calculate_intersection_percentage(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1[:4]
    x3, y3, x4, y4 = bbox2[:4]

    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)

    intersection_area = max(0, x6 - x5) * max(0, y6 - y5)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    return intersection_area / (area1 + area2 - intersection_area)

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        print(f"Loaded: {image_path}")
        return image
    else:
        raise Exception(f"Failed to load: {image_path}")    
    
delete_files_in_folder(EXPORT_FOLDER_PATH)
yolo_model = YOLOWrapper(input("Enter the path to the model: "))
sam2_model = SAMWrapper(Path(__file__).resolve().parent.parent / "trained_yolo_models" / "sam2_s.pt")
image_paths = return_image_paths(read_folder)

for image_path in image_paths:
    image = load_image(image_path)

    yolo_model.detect_frame(frame = image, confidence_threshold = DETECTOR_CONFIDENCE_THRESHOLD)    
    #yolo_model.draw_recent_bbox_on_frame(frame = image)

    detection_bboxes = []
    for detection_no, detection in enumerate(yolo_model.get_recent_detection_results()["normalized_bboxes"]):
        for previous_detection_bbox in detection_bboxes:
            if calculate_intersection_percentage(detection, previous_detection_bbox) > 0.5: # To prevent overlapping detections
                break
        else:
            detection_bboxes.append(detection)
        
    segmented_bboxes = []
    for detection_bbox in detection_bboxes:
        segment_bbox = sam2_model.segment_by_rect(frame = image, normalized_bbox_coordinates = detection_bbox[:4], class_name = detection_bbox[5])
        segment_bbox.extend(detection_bbox[4:])
        segmented_bboxes.append(segment_bbox)
    
    label_file_path = EXPORT_FOLDER_PATH / f"{Path(image_path).stem}.txt"

    if SHOW_ZOOMED_PERSON:
        cv2.imshow("Person Head Segments", image)
        key = cv2.waitKey(WAIT_SHOW_ZOOMED_PERSON) & 0xFF
        if key == 27:  # ESC key to break the loop
            break
        elif key == ord('s'):
            print(f"Saving: {label_file_path}")
            with open(label_file_path, 'w') as label_file:
                for segmented_bbox in segmented_bboxes:
                    class_id = int(segmented_bbox[6])
                    xcn, ycn, wn, hn = (segmented_bbox[0] + segmented_bbox[2]) / 2 / image.shape[1], (segmented_bbox[1] + segmented_bbox[3]) / 2 / image.shape[0], (segmented_bbox[2] - segmented_bbox[0]) / image.shape[1], (segmented_bbox[3] - segmented_bbox[1]) / image.shape[0] 
                    label_file.write(f"{class_id} {xcn} {ycn} {wn} {hn}\n")
            
        
cv2.destroyAllWindows()