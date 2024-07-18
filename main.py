import av
import numpy as np
import cv2
import json
from ultralytics import YOLO

# Function to process video and classify frames
def process_video(video_path, model, class_names):
    container = av.open(video_path)
    class_timestamps = {class_name: [] for class_name in class_names}
    
    ans = [[], []]
    for frame in container.decode(video=0):
        
        # Convert frame to ndarray
        frame_array = frame.to_ndarray(format='bgr24')

        # Preprocess the frame for the model
        frame_resized = cv2.resize(frame_array, (640, 640))  # YOLO expects 640x640 input size
        results = model.predict(source=frame_resized, imgsz=640)

        # Parse results and update class timestamps
        for result in results:
            count_0 = 0
            count_1 = 0
            for detection in result:
                # class_id = detection[5]  # Class ID
                # print(detection.boxes.cls.tolist())
                p = detection.boxes.cls.tolist()
                if p[0] == 0:
                    count_0 += 1
                if p[0] == 1:
                    count_1 += 1
                        
            timestamp = frame.time
            if count_0 >= 1: ans[0].append(round(timestamp, 2))
            if count_1 >= 1: ans[1].append(round(timestamp, 2))
            
    # Save the results to a JSON file
    with open('timestamps.json', 'w') as json_file:
        res = {
            "Pepsi_pts": ans[1],
            "Cocacols_pts": ans[0]
        }
        json.dump(res, json_file)

# Load your YOLO model
model = YOLO("best.pt")

# Define the class names (adjust as needed)
class_names = ['cococola', 'pepsico'] 

# Path to the input video file
video_path = 'video.mp4'

# Process the video and save results
process_video(video_path, model, class_names)
