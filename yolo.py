# TODO: Nudge the algo to expand width first, then height
# TODO: Force a min height but also a min width. Area needs to be *portrait* or close to it. Not slim.


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from shapely.geometry import Polygon

# Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the image
img = cv2.imread('6.png')
final_img = img.copy()

# # Perform object detection on the image
# results = model(img)
# res = results.pandas().xyxy[0]

# for obj in results.xyxy[0]:
#     x1, y1, x2, y2, conf, cls = obj[:6]
#     label = model.names[int(cls)]
#     print(
#         f'Found {label} with confidence {conf:.2f} at ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})')
#     # Remove the detected objects from the image by drawing a black rectangle around them
#     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)

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

# show mask image
# cv2.imshow("mask", ~mask)
# cv2.waitKey(0)


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


def find_top_k_rectangles(final_img, mask, k, max_zero_percentage=0.0, aspect_ratio_threshold=4):

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
                    # cv2.rectangle(final_img, (x, start),
                    #               (x, y - 1), (0, 0, 0), -1)
                    start = None
        if start is not None:
            regions.append((start, mask.shape[0] - 1))
            # cv2.rectangle(final_img, (x, start), (x, y - 1), (0, 0, 0), -1)

        if not regions:
            # If the column has no runs of 255 pixels, add a region spanning the entire column
            regions.append((0, mask.shape[0] - 1))
            # cv2.rectangle(final_img, (x, start), (x, y - 1), (0, 0, 0), -1)

        col_regions.append(regions)

    # Merge the row and column regions into rectangular regions
    regions = []
    for x, col in enumerate(col_regions):
        for y1, y2 in col:
            for x1, x2 in row_regions[y1]:
                if x1 <= x <= x2:
                    # Check if any pixel within the candidate rectangle has a value of 0 in the original binary mask
                    if np.sum(mask[y1:y2+1, x1:x2+1] == 0) / ((y2 - y1 + 1) * (x2 - x1 + 1)) > max_zero_percentage:
                        continue
                    # Compute the area and aspect ratio of the rectangle
                    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    aspect_ratio = (x2 - x1 + 1) / (y2 - y1 + 1)
                    # Filter out rectangles with aspect ratio above the threshold or below a minimum value
                    if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1/aspect_ratio_threshold:
                        continue
                    regions.append((x1, y1, x2 - x1 + 1, y2 - y1 + 1))

                    # Compute the area of the rectangle and its union with previously selected rectangles
                    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    overlap_area = sum([rect_overlap_area(
                        (x1, y1, x2 - x1 + 1, y2 - y1 + 1), prev_rect) for prev_rect in regions])
                    # Check if the rectangle is at least 50% unique
                    if rect_area / (rect_area + overlap_area) < 0.5:
                        continue
                    regions.append((x1, y1, x2 - x1 + 1, y2 - y1 + 1))

    for y, row in enumerate(row_regions):
        for x1, x2 in row:
            for y1, y2 in col_regions[x1]:
                if y1 <= y <= y2:
                    # Check if any pixel within the candidate rectangle has a value of 0 in the original binary mask
                    if np.sum(mask[y1:y2+1, x1:x2+1] == 0) / ((y2 - y1 + 1) * (x2 - x1 + 1)) > max_zero_percentage:
                        continue
                    # Compute the area and aspect ratio of the rectangle
                    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    aspect_ratio = (x2 - x1 + 1) / (y2 - y1 + 1)
                    # Filter out rectangles with aspect ratio above the threshold or below a minimum value
                    if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1/aspect_ratio_threshold:
                        continue
                    regions.append((x1, y1, x2 - x1 + 1, y2 - y1 + 1))

                    # Compute the area of the rectangle and its union with previously selected rectangles
                    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    overlap_area = sum([rect_overlap_area(
                        (x1, y1, x2 - x1 + 1, y2 - y1 + 1), prev_rect) for prev_rect in regions])
                    # Check if the rectangle is at least 50% unique
                    # if rect_area / (rect_area + overlap_area) < 0.5:
                    #     continue
                    regions.append((x1, y1, x2 - x1 + 1, y2 - y1 + 1))

    # Sort the regions by area in descending order
    regions = sorted(
        regions, key=lambda region: region[2] * region[3], reverse=True)

   # Keep the top k regions
    unique_regions = []
    for region in regions:
        if not any(rect_overlap_area(region, prev_rect) >= 0.5 * min(region[2] * region[3], prev_rect[2] * prev_rect[3]) for prev_rect in unique_regions):
            unique_regions.append(region)
        if len(unique_regions) == k:
            break

    # Draw rectangles around the top k regions
    # TODO EDIT TO FIND UNIQUES
    for x, y, w, h in regions:
        cv2.rectangle(final_img, (x, y), (x + w - 1,
                      y + h - 1), (0, 0, 0), 2)

    # Get the x, y coordinates of the top k regions
    coords = [(x, y) for x, y, _, _ in unique_regions]

    return coords


def is_eligible_top_k_rectangle(mask, coords, max_zero_percentage=0.00, aspect_ratio_threshold=1):
    # Extract the coordinates of the rectangle
    x1, y1, x2, y2 = coords

    # Check if any pixel within the rectangle has a value of 0 in the original binary mask
    rect_mask = mask[y1:y2+1, x1:x2+1]
    zero_count = np.count_nonzero(rect_mask == 0)
    rect_size = rect_mask.size
    if zero_count / rect_size > max_zero_percentage:
        # If the percentage of pixels with a value of 0 in the rectangle exceeds the maximum allowed value,
        # highlight the problematic pixels in red and return False
        rect_mask[rect_mask == 0] = 128
        rect_mask[rect_mask == 255] = 0
        return False

    # Compute the area and aspect ratio of the rectangle
    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    aspect_ratio = (x2 - x1 + 1) / (y2 - y1 + 1)

    # Check if the aspect ratio of the rectangle exceeds the threshold or is below the minimum value
    if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1/aspect_ratio_threshold:
        # If the aspect ratio is outside the allowed range, highlight the rectangle in blue and return False
        rect_mask[rect_mask == 0] = 128
        rect_mask[rect_mask == 255] = 0
        return False

    # If the rectangle is eligible, return True
    return True


def find_top_k_rectangles_both(final_img, mask, k, aspect_ratio_threshold=4, max_zero_percentage=0.15):
    def get_regions(axis):
        regions = []
        start = None
        for i in range(mask.shape[axis]):
            indices = [i, slice(None)] if axis == 0 else [slice(None), i]
            if np.any(mask[indices] == 255):
                if start is None:
                    start = i
            elif start is not None:
                regions.append((start, i - 1))
                start = None
        if start is not None:
            regions.append((start, mask.shape[axis] - 1))
        return regions

    row_regions = get_regions(0)
    col_regions = get_regions(1)

    regions = []
    for y, row in enumerate(row_regions):
        for x1, x2 in row:
            for y1, y2 in col_regions:
                if y1 <= y <= y2:
                    # Check if any pixel within the rectangle has a value of 0 in the original binary mask
                    if np.sum(mask[y1:y2+1, x1:x2+1] == 0) / ((y2 - y1 + 1) * (x2 - x1 + 1)) > max_zero_percentage:
                        continue
                    # Compute the area and aspect ratio of the rectangle
                    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    aspect_ratio = (x2 - x1 + 1) / (y2 - y1 + 1)
                    # Filter out rectangles with aspect ratio above the threshold or below a minimum value
                    if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1/aspect_ratio_threshold:
                        continue
                    regions.append((x1, y1, x2 - x1 + 1, y2 - y1 + 1))

                    # Compute the area of the rectangle and its union with previously selected rectangles
                    rect_area = (x2 - x1 + 1) * (y2 - y1 + 1)
                    overlap_area = sum([rect_overlap_area(
                        (x1, y1, x2 - x1 + 1, y2 - y1 + 1), prev_rect) for prev_rect in regions])
                    # Check if the rectangle is at least 50% unique
                    if rect_area / (rect_area + overlap_area) < 0.5:
                        continue
                    regions.append((x1, y1, x2 - x1 + 1, y2 - y1 + 1))

    # Sort the regions by area in descending order
    regions = sorted(
        regions, key=lambda region: region[2] * region[3], reverse=True)

    # Keep the top k regions
    top_k_regions = []
    for region in regions:
        if not any(rect_overlap_area(region, prev_rect) >= 0.5 * min(region[2] * region[3], prev_rect[2] * prev_rect[3]) for prev_rect in top_k_regions):
            top_k_regions.append(region)
        if len(top_k_regions) >= k:
            break

        # Draw rectangles around the top k regions
    for x, y, w, h in top_k_regions:
        cv2.rectangle(final_img, (x, y), (x + w - 1,
                      y + h - 1), (255, 255, 255), 2)

    # Get the x, y coordinates of the top k regions
    coords = [(x, y) for x, y, _, _ in top_k_regions]

    return top_k_regions


# Find the top 5 largest rectangular regions not covered by the bounding boxes
coords = find_top_k_rectangles(final_img, ~mask, 3, aspect_ratio_threshold=6, max_zero_percentage=0.0)
# coords = 0, 0, 500, 250
# res = is_eligible_top_k_rectangle(~mask, coords)
print("coordinates of top 5 largest rectangular regions not covered by the bounding boxes:", coords)

# show final image
cv2.imshow("mask", mask)
cv2.imshow("final image", final_img)
cv2.waitKey(0)

quit()
