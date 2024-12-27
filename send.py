import cv2
import time
import torch
import socket
import struct
import numpy as np
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from models.yolo import Model

def preprocess_image(img, img_size=640):
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    top_pad, bottom_pad = (img_size - new_h) // 2, img_size - new_h - (img_size - new_h) // 2
    left_pad, right_pad = (img_size - new_w) // 2, img_size - new_w - (img_size - new_w) // 2
    img_resized = cv2.copyMakeBorder(img_resized, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT)
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)
    img_resized = np.ascontiguousarray(img_resized)
    img_resized = torch.from_numpy(img_resized).float() / 255.0
    if img_resized.ndimension() == 3:
        img_resized = img_resized.unsqueeze(0)
    return img_resized

def detect_humans(image, model, device, img_size=640, conf_thresh=0.25, iou_thresh=0.45):
    img_preprocessed = preprocess_image(image, img_size).to(device)
    if next(model.parameters()).dtype == torch.float16:
        img_preprocessed = img_preprocessed.half()
    with torch.no_grad():
        pred = model(img_preprocessed)[0]
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=[0])
    return pred




def plot_detections(image, predictions, img_size=640):
    img_copy = image.copy()
    for det in predictions:
        if len(det):
            det[:, :4] = scale_coords((img_size, img_size), det[:, :4], img_copy.shape).round()
            for *xyxy, conf, cls in det:
                cv2.rectangle(img_copy, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(img_copy, f'Person {conf:.2f}', (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return img_copy

def calculate_bbox_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2

def calculate_direction(center_x, center_y, frame_center_x, frame_center_y, x1, y1, x2, y2):
    instructions = []
    if x1 <= frame_center_x <= x2 and y1 <= frame_center_y <= y2:
        instructions.append("Centered")
    else:
        if frame_center_x < x1:
            instructions.append("Move Right")
        elif frame_center_x > x2:
            instructions.append("Move Left")
        if frame_center_y < y1:
            instructions.append("Move Forward")
        elif frame_center_y > y2:
            instructions.append("Move Backward")
    return ", ".join(instructions)

def main():
    device = select_device('0' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # Load model
    model = torch.load('yolov7.pt', map_location=device)['model']
    model.eval()
    print("Model loaded successfully")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 65432))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # print("Frame captured successfully")
        
        frame_height, frame_width = frame.shape[:2]
        frame_center_x, frame_center_y = frame_width // 2, frame_height // 2
        predictions = detect_humans(frame, model, device)
        print(predictions)
        num_people, closest_person, closest_distance = 0, None, float('inf')
        
        for det in predictions:
            for *xyxy, conf, cls in det:
                if conf.item() >= 0.4:
                    num_people += 1
                    x1, y1, x2, y2 = map(int, xyxy)
                    bbox_center_x, bbox_center_y = calculate_bbox_center(x1, y1, x2, y2)
                    distance = ((bbox_center_x - frame_center_x) ** 2 + (bbox_center_y - frame_center_y) ** 2) ** 0.5
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_person = (bbox_center_x, bbox_center_y, x1, y1, x2, y2)

        # print(f"Number of people detected: {num_people}")  # Debugging print statement

        if closest_person:
            bbox_center_x, bbox_center_y, x1, y1, x2, y2 = closest_person
            direction = calculate_direction(bbox_center_x, bbox_center_y, frame_center_x, frame_center_y, x1, y1, x2, y2)
            print(f"Closest Person at ({bbox_center_x}, {bbox_center_y}): {direction}")
            client_socket.sendall(f"Human Detected bbox=({bbox_center_x}, {bbox_center_y})".encode())
        else:
            print("No humans detected.")
            client_socket.sendall(f"no_human".encode())

        frame_with_detections = plot_detections(frame, predictions)

        if frame_with_detections is not None:
            print("Displaying frame with detections")
            cv2.imshow('YOLOv7 Human Detection', frame_with_detections)
        else:
            print("Error displaying the frame.")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting live feed")
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
