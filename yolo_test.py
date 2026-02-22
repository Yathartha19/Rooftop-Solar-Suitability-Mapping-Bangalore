from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO('runs/segment/train/weights/best.pt')
results = model("test_image.png", conf=0.3)

img = cv2.imread("test_image.png")

masks = results[0].masks
if masks:
    for mask in masks.data.cpu().numpy():
        binary_mask = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(img, contours, -1, (0, 0, 200, 0.4), thickness=cv2.FILLED)  

        cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=1)  

output_path = "output_with_masks.png"
cv2.imwrite(output_path, img)
print(f"✅ Saved output to {output_path}")
