import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from shapely.geometry import Polygon


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the image
img = cv2.imread('image.jpeg')
final_img = img.copy()

# Perform object detection on the image
results = model(img)
res = results.pandas().xyxy[0]

for obj in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = obj[:6]
    label = model.names[int(cls)]
    print(
        f'Found {label} with confidence {conf:.2f} at ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})')
    # Remove the detected objects from the image by drawing a black rectangle around them
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)

# Load the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform adaptive thresholding to obtain a binary image
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8)

# Find contours in the binary image
contours, _ = cv2.findContours(
    adaptive_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Find the top 5 largest contours in the image
contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                  reverse=True)[1:6]
print("number of contours: ", len(contours))

# Find the bounding boxes of the top 5 largest contours in the image
bounding_boxes = []
for c in sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:5]:
    x, y, w, h = cv2.boundingRect(c)
    bounding_boxes.append((x, y, w, h))

# Draw the top 5 contours on the original image
for i, c in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    print(x, y, w, h)
    print("size of each contur in image: ", cv2.contourArea(c))
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, f"#{i+1}", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Create a mask that marks the areas covered by the bounding boxes
mask = np.zeros_like(gray)
for x, y, w, h in bounding_boxes:
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)


def find_top_k_rectangles(final_img, mask, k, aspect_ratio_threshold=4, max_zero_percentage=0.15):
    def rect_overlap_area(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        left = max(x1, x2)
        right = min(x1 + w1 - 1, x2 + w2 - 1)
        top = max(y1, y2)
        bottom = min(y1 + h1 - 1, y2 + h2 - 1)
        if right >= left and bottom >= top:
            return (right - left + 1) * (bottom - top + 1)
        else:
            return 0

    # Find the rectangular regions in each row
    row_regions = []
    for y in range(mask.shape[0]):
        regions = []
        start = None
        for x in range(mask.shape[1]):
            if mask[y, x] == 255:
                if start is None:
                    start = x
            else:
                if start is not None:
                    regions.append((start, x - 1))
                    start = None
        if start is not None:
            regions.append((start, mask.shape[1] - 1))
        row_regions.append(regions)

    # Find the rectangular regions in each column
    col_regions = []
    for x in range(mask.shape[1]):
        regions = []
        start = None
        for y in range(mask.shape[0]):
            if mask[y, x] == 255:
                if start is None:
                    start = y
            else:
                if start is not None:
                    regions.append((start, y - 1))
                    start = None
        if start is not None:
            regions.append((start, mask.shape[0] - 1))
        col_regions.append(regions)

    # Merge the row and column regions into rectangular regions
    regions = []
    for y, row in enumerate(row_regions):
        for x1, x2 in row:
            for y1, y2 in col_regions[x1]:
                if y1 <= y <= y2:
                    indices = (slice(y1, y2 + 1), slice(x1, x2 + 1))
                    if np.any(mask[indices] == 255):
                        continue
                    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    aspect_ratio = (x2 - x1 + 1) / (y2 - y1 + 1)
                    if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1/aspect_ratio_threshold:
                        continue
                    rect = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
                    overlap_area = sum(
                        [rect_overlap_area(rect, prev_rect) for prev_rect in regions])
                    if rect_area / (rect_area + overlap_area) < 0.5:
                        continue
                    regions.append(rect)

    # Merge the column and row regions into rectangular regions
    for x, col in enumerate(col_regions):
        for y1, y2 in col:
            for x1, x2 in row_regions[y1]:
                if x1 <= x <= x2:
                    indices = (slice(y1, y2 + 1), slice(x1, x2 + 1))
                    if np.any(mask[indices] == 255):
                        continue
                    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    aspect_ratio = (x2 - x1 + 1) / (y2 - y1 + 1)
                    if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1/aspect_ratio_threshold:
                        continue
                    rect = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
                    overlap_area = sum(
                        [rect_overlap_area(rect, prev_rect) for prev_rect in regions])
                    if rect_area / (rect_area + overlap_area) < 0.5:
                        continue
                    regions.append(rect)

    row_regions = []

    for y in range(mask.shape[0]):
        regions = []
        start = None
        for x in range(mask.shape[1]):
            if mask[y, x] == 255:
                if start is None:
                    start = x
            else:
                if start is not None:
                    regions.append((start, x - 1))
                    start = None
        if start is not None:
            regions.append((start, mask.shape[1] - 1))
        row_regions.append(regions)

    # Find the rectangular regions in each column
    col_regions = []
    for x in range(mask.shape[1]):
        regions = []
        start = None
        for y in range(mask.shape[0]):
            if mask[y, x] == 255:
                if start is None:
                    start = y
            else:
                if start is not None:
                    regions.append((start, y - 1))
                    start = None
        if start is not None:
            regions.append((start, mask.shape[0] - 1))
        col_regions.append(regions)

    # Merge the row and column regions into rectangular regions
    regions = []
    for y, row in enumerate(row_regions):
        for x1, x2 in row:
            for y1, y2 in col_regions[x1]:
                if y1 <= y <= y2:
                    indices = (slice(y1, y2 + 1), slice(x1, x2 + 1))
                    if np.any(mask[indices] == 255):
                        continue
                    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    aspect_ratio = (x2 - x1 + 1) / (y2 - y1 + 1)
                    if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1/aspect_ratio_threshold:
                        continue
                    rect = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
                    overlap_area = sum([rect_overlap_area(rect, prev_rect)
                                        for prev_rect in regions])
                    if rect_area / (rect_area + overlap_area) < 0.5:
                        continue
                    regions.append(rect)

    # Merge the column and row regions into rectangular regions
    for x, col in enumerate(col_regions):
        for y1, y2 in col:
            for x1, x2 in row_regions[y1]:
                if x1 <= x <= x2:
                    indices = (slice(y1, y2 + 1), slice(x1, x2 + 1))
                    if np.any(mask[indices] == 255):
                        continue
                    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    aspect_ratio = (x2 - x1 + 1) / (y2 - y1 + 1)
                    if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1/aspect_ratio_threshold:
                        continue
                    rect = (x1, y1, x2 - x1 + 1, y2 - y1 + 1)
                    overlap_area = sum([rect_overlap_area(rect, prev_rect)
                                        for prev_rect in regions])
                    if rect_area / (rect_area + overlap_area) < 0.5:
                        continue
                    regions.append(rect)

    # Remove regions with high percentage of zeros
    new_regions = []
    for rect in regions:
        indices = (slice(rect[1], rect[1] + rect[3]),
                   slice(rect[0], rect[0] + rect[2]))
        img_slice = final_img[indices]
        zero_percentage = np.count_nonzero(img_slice == 0) / img_slice.size
        if zero_percentage > max_zero_percentage:
            continue
        new_regions.append(rect)
    regions = new_regions

    # Select the top k regions based on the area again after removing the high percentage of zeros
    regions = sorted(regions, key=lambda x: -x[2] * x[3])[:k]

    return regions


# Find the top 5 largest rectangular regions not covered by the bounding boxes
coords = find_top_k_rectangles(final_img, ~mask, 20)
print("coordinates of top 5 largest rectangular regions not covered by the bounding boxes:", coords)

# show final image
cv2.imshow("final image", final_img)
cv2.waitKey(0)

quit()
