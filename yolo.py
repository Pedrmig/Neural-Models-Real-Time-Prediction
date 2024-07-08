from ultralytics import YOLO
import cv2
import math

# Initialize the camera and set the image resolution
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the YOLO model
model = YOLO('yolo-Weights/yolov8n.pt')

# Define the classes to be detected
classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Video capture loop
while True:
    success, img = cap.read() # Read the image from the camera
    results = model(img, stream=True) # Send the image to the YOLO model
    
    for r in results:
        boxes = r.boxes # Extract the boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0] # Extract the coordinates
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) # Convert to integers
            
            # Draw the box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Calculate the confidence of the detected object
            confidence = math.ceil(box.conf[0]*100)
            print(confidence)
            
            # Detect the name
            cls = int(box.cls[0])
            print(classNames[cls])
            
            # Draw the name of the detected object
            org = (x1, y1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"{classNames[cls]} {confidence:2f}", org, font, 1, (255, 0, 0), 2)
            
    # Create the display window
    cv2.imshow("Webcam", img)
    # Exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to exit
        break
    
cap.release()
cv2.destroyAllWindows()
