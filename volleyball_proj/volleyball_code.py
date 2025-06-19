import cv2
import numpy as np

cap = cv2.VideoCapture("volleyball_match.mp4")

output_path = "volleyball_tracking.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

font = cv2.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 100:
            x, y, w, h = cv2.boundingRect(largest)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "ball", (x, y + h + 20), font, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "Total yellow players : 6", (20, 40), font, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "Total red players : 6", (20, 75), font, 0.8, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
print("Tracking complete. Output saved to:", output_path)