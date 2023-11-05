<<<<<<< HEAD
import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow model
model = tf.saved_model.load('saved_model')

# Load labels from the COCO dataset
LABELS_PATH = 'mscoco_label_map.pbtxt'

def load_coco_labels(label_file):
    labels = {}
    with open(label_file, 'r') as file:
        lines = file.read().splitlines()
        for i in range(0, len(lines), 5):
            label_index = int(lines[i + 2].split(': ')[1])
            label_name = lines[i + 3].split(': ')[1].strip('"')
            labels[label_index] = label_name
    return labels

labels = load_coco_labels(LABELS_PATH)

def detect_objects(frame):
    # Convert the frame color from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to a tensor and expand dimensions
    input_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Pass the tensor to the model for detections
    detections = model(input_tensor)
    
    return detections

cap = cv2.VideoCapture(0)  # 0 for default camera

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (300, 300))
        detections = detect_objects(resized_frame)

        detection_scores = detections['detection_scores'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy().astype(int)
        detection_boxes = detections['detection_boxes'][0].numpy()

        height, width, _ = frame.shape
        for score, cls, box in zip(detection_scores, detection_classes, detection_boxes):
            if score > 0.5:  # Threshold
                label = labels.get(cls, "Unknown")
                box = box * [height, width, height, width]
                y1, x1, y2, x2 = map(int, box)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                frame = cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Face and Object Recognition', frame)

        # Stop the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
=======
import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow model
model = tf.saved_model.load('saved_model')

# Load labels from the COCO dataset
LABELS_PATH = 'mscoco_label_map.pbtxt'

def load_coco_labels(label_file):
    labels = {}
    with open(label_file, 'r') as file:
        lines = file.read().splitlines()
        for i in range(0, len(lines), 5):
            label_index = int(lines[i + 2].split(': ')[1])
            label_name = lines[i + 3].split(': ')[1].strip('"')
            labels[label_index] = label_name
    return labels

labels = load_coco_labels(LABELS_PATH)

def detect_objects(frame):
    # Convert the frame color from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to a tensor and expand dimensions
    input_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # Pass the tensor to the model for detections
    detections = model(input_tensor)
    
    return detections

cap = cv2.VideoCapture(0)  # 0 for default camera

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        resized_frame = cv2.resize(frame, (300, 300))
        detections = detect_objects(resized_frame)

        detection_scores = detections['detection_scores'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy().astype(int)
        detection_boxes = detections['detection_boxes'][0].numpy()

        height, width, _ = frame.shape
        for score, cls, box in zip(detection_scores, detection_classes, detection_boxes):
            if score > 0.5:  # Threshold
                label = labels.get(cls, "Unknown")
                box = box * [height, width, height, width]
                y1, x1, y2, x2 = map(int, box)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                frame = cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Face and Object Recognition', frame)

        # Stop the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
>>>>>>> f6de5a9ff2d56b35461eaf781ac04c2fa6fc844d
