import os, copy, cv2, pprint
from pathlib import Path
from ultralytics import YOLO, SAM
import numpy as np

# Paths to the folders
EXPORT_FOLDER_PATH = Path(__file__).resolve().parent.parent / "exports" # where labels will be saved

SHOW_ZOOMED_PERSON = True
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
            "normalized_bboxes": [], # List of normalized bounding boxes in the format [x1n, y1n, x2n, y2n, bbox_confidence, class_name]
            "keypoints": [] # List of keypoints in the format [x, y, confidence]
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
    
    def pose_frame(self, frame:np.ndarray = None, confidence_threshold:float = POSE_DETECTOR_CONFIDENCE_THRESHOLD) -> None:
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
    
    def segment_by_rect(self, frame:np.ndarray = None, bbox_coordinates:list = None) -> None:
        self.__clear_recent_segmentation_results()
        
        segmentation_results = self.sam_object(frame, bboxes=[bbox_coordinates], show=False)[0]

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

        cv2.rectangle(frame, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)   
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

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        print(f"Loaded: {image_path}")
        return image
    else:
        raise Exception(f"Failed to load: {image_path}")    
    
delete_files_in_folder(EXPORT_FOLDER_PATH)
pose_model = YOLOWrapper(Path(__file__).resolve().parent.parent / "trained_yolo_models" / "yolov8x-pose.pt")
sam2_model = SAMWrapper(Path(__file__).resolve().parent.parent / "trained_yolo_models" / "sam2_s.pt")
image_paths = return_image_paths(read_folder)

for image_path in image_paths:
    image = load_image(image_path)

    pose_model.pose_frame(frame = image, confidence_threshold = POSE_DETECTOR_CONFIDENCE_THRESHOLD)    

    segmentation_centers = []
    for person_no, person_keypoints in enumerate(pose_model.get_recent_detection_results()["keypoints"]):
        HEAD_RELATED_KEYPOINTS = ["nose", "right_eye", "left_eye", "left_ear", "right_ear"]
        
        valid_xyns = [(xn, yn) for keypoint_name in HEAD_RELATED_KEYPOINTS if (xn := person_keypoints[pose_model.get_keypoint_index_by_name(keypoint_name)][0]) > 0 and (yn := person_keypoints[pose_model.get_keypoint_index_by_name(keypoint_name)][1]) > 0]
        if len(valid_xyns) > 0:
            mean_xn = sum([xn for xn, yn in valid_xyns]) / len(valid_xyns)
            mean_yn = sum([yn for xn, yn in valid_xyns]) / len(valid_xyns)

            segmentation_centers.append((int(mean_xn*image.shape[1]), int(mean_yn*image.shape[0]-10) ))
        
    
    PARAM_SEGMENTATION_BBOX_SIZE = 100
    segmented_bboxes = []
    for segmentation_center in segmentation_centers:
        segmentation_applied_bbox = [
            max(0, segmentation_center[0]-PARAM_SEGMENTATION_BBOX_SIZE//2),
            max(0, segmentation_center[1]-PARAM_SEGMENTATION_BBOX_SIZE//2),
            min(image.shape[1], segmentation_center[0]+PARAM_SEGMENTATION_BBOX_SIZE//2),
            min(image.shape[0], segmentation_center[1]+PARAM_SEGMENTATION_BBOX_SIZE//2),
        ]
        segment_bbox = sam2_model.segment_by_rect(frame = image, bbox_coordinates = segmentation_applied_bbox)
        segmented_bboxes.append(segment_bbox)

    DEFAULT_CLASS_NO = 0
    label_file_path = EXPORT_FOLDER_PATH / f"{Path(image_path).stem}.txt"
    with open(label_file_path, 'w') as label_file:
        for segmented_bbox in segmented_bboxes:
            class_id = int(DEFAULT_CLASS_NO)
            xcn, ycn, wn, hn = (segmented_bbox[0] + segmented_bbox[2]) / 2 / image.shape[1], (segmented_bbox[1] + segmented_bbox[3]) / 2 / image.shape[0], (segmented_bbox[2] - segmented_bbox[0]) / image.shape[1], (segmented_bbox[3] - segmented_bbox[1]) / image.shape[0] 
            label_file.write(f"{class_id} {xcn} {ycn} {wn} {hn}\n")

    if SHOW_ZOOMED_PERSON:
        cv2.imshow("Person Head Segments", image)
        key = cv2.waitKey(WAIT_SHOW_ZOOMED_PERSON) & 0xFF
        if key == 27:  # ESC key to break the loop
            break

        
      
  
cv2.destroyAllWindows()