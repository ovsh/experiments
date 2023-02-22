# TODO: Nudge the algo to expand width first, then height
# TODO: Force a min height but also a min width. Area needs to be *portrait* or close to it. Not slim.


import matplotlib.pyplot as plt
import math
import keras_ocr
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from shapely.geometry import Polygon


#define python class that refactors the code in this page 
#

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

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

def add_text_to_image(text, raw_img, coords, padding=0.95, centered=True, split_lines=True):
    # Convert the input image to the PIL format
    raw_img = cv2.convertScaleAbs(raw_img)
    img = Image.fromarray(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Get the size of the bounding box
    x1, y1, x2, y2 = map(int, coords)
    width, height = x2 - x1, y2 - y1

    # Calculate the maximum font size
    font_size = 1
    font = ImageFont.truetype("arial.ttf", font_size)

    # Calculate the maximum font size for single line text
    while font.getlength(text) < padding * width and font.getbbox(text)[3] - font.getbbox(text)[1] < padding * height:
        font_size += 1
        font = ImageFont.truetype("arial.ttf", font_size)
    # font_size -= 1  # shrink font once because it exceeded the bounding box that we set in the above loop

    # Calculate the maximum font size for multi-line text
    if split_lines:
        # Create the ImageFont object for multi-line text
        font_size_multi_line = 1
        font_multi_line = ImageFont.truetype("arial.ttf", font_size_multi_line)

        # Split the text into two lines that fit within the bounding box
        max_chars_per_line = len(text) // 2  # split text in half
        while True:
            # Calculate the size of the first line
            first_line = text[:max_chars_per_line]
            first_line_size = font_multi_line.getsize(first_line)

            # Check if the first line fits within the bounding box
            if first_line_size[0] <= padding * width and first_line_size[1] <= padding * height:
                # Calculate the size of the second line
                second_line = text[max_chars_per_line:]
                second_line_size = font_multi_line.getsize(second_line)

                # Check if the second line fits within the bounding box
                if second_line_size[0] <= padding * width and second_line_size[1] <= padding * height:
                    # Both lines fit, so increase font size and continue
                    font_size_multi_line += 1
                    font_multi_line = ImageFont.truetype(
                        "arial.ttf", font_size_multi_line)
                else:
                    print('break condition hit. here is the size of the font: {}, the width: {}, and the height: {}'.format(
                        font_size_multi_line, padding * width, padding * height))
                    # Second line does not fit, so use previous font size
                    break
            else:
                print('break condition hit. here is the size of the font: {}, the width: {}, and the height: {}'.format(
                    font_size_multi_line, padding * width, padding * height))
                # First line does not fit, so use previous font size
                break

        # Set font size to be the minimum of single line and multi-line font sizes
        if split_lines:
            # Set font size to be the minimum of single line and multi-line font sizes
            font_size = font_size_multi_line - 1
        else:
            # Use single-line font size
            font_size = font_size
    font = ImageFont.truetype("arial.ttf", font_size)

    if split_lines:
        # Split the text into two lines if necessary
        if font.getsize(text)[0] > padding * width:
            # Find the midpoint of the text and split it into two lines
            midpoint = len(text) // 2
            first_line = text[:midpoint].strip()
            second_line = text[midpoint:].strip()

            # Add the lines to the list of lines
            lines = [first_line, second_line]
        else:
            # Split the text into multiple lines
            words = text.split()
            lines = []
            line = ""
            for word in words:
                if font.getlength(line + word) < padding * width:
                    line += word + " "
                else:
                    lines.append(line[:-1])
                    line = word + " "
            lines.append(line[:-1])
    else:
        # Split the text into multiple lines
        words = text.split()
        lines = []
        line = ""
        for word in words:
            if font.getlength(line + word) < padding * width:
                line += word + " "
            else:
                lines.append(line[:-1])
                line = word + " "
        lines.append(line[:-1])

    # Calculate the coordinates of the top-left corner of the text
    if centered:
        x = x1 + (width - font.getbbox(lines[0])
                  [2] + font.getbbox(lines[0])[0]) / 2
        y = y1 + (height - font_size * len(lines)) / 2
    else:
        x = x1 + (1-padding)*0.5 * width
        y = y1 + (1-padding)*0.5 * height

    # Draw the text on the image
    for line in lines:
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        print("Text drawn at ({}, {}). Text = {}".format(x, y, line))
        y += font_size

    # Print where you drew the text, including text font size and width and height
    print("Text drawn at ({}, {})".format(x, y))
    print("Text font size: {}".format(font_size))
    print("Text width: {}".format(font.getbbox(
        lines[0])[2] - font.getbbox(lines[0])[0]))

    # Convert the modified image back to the OpenCV format
    modified_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # draw a bounding box around the original coords in black and then draw another bounding box around the bounding box of the text in red
    cv2.rectangle(modified_image, (x1, y1), (x2, y2), (0, 0, 0), 2)
    # cv2.rectangle(modified_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    return modified_image


def find_top_k_rectangles_new(final_img, mask, k, max_zero_percentage=0.0, aspect_ratio_threshold=4):
    regions = []
    image_size = mask.shape[0] * mask.shape[1]

    # Expand from a random starting point
    def expand_from_point(x, y):
        left, right = x, x
        while left > 0 and mask[y, left-1] == 255:
            left -= 1
        while right < mask.shape[1]-1 and mask[y, right+1] == 255:
            right += 1
        top, bottom = y, y
        while top > 0 and mask[top-1, x] == 255:
            top -= 1
        while bottom < mask.shape[0]-1 and mask[bottom+1, x] == 255:
            bottom += 1
        return (left, top, right-left+1, bottom-top+1)

    # Binary search to find the largest rectangle from a starting point
    def find_largest_rect(x, y):
        best_rect = expand_from_point(x, y)
        max_width = min(best_rect[2], best_rect[3]*aspect_ratio_threshold)
        min_width = max(1, best_rect[3]/aspect_ratio_threshold)
        while max_width >= min_width:
            width = (max_width + min_width) // 2
            height = width / aspect_ratio_threshold
            rect = expand_from_point(x, y)
            if rect[2] < width or rect[3] < height:
                max_width = width - 1
            else:
                best_rect = rect
                min_width = width + 1
        return best_rect

    # Iterate over the entire image
    for i in range(300):
        if len(regions) >= k:
            break
        x, y = random.randint(
            0, mask.shape[1]-1), random.randint(0, mask.shape[0]-1)
        if mask[y, x] == 255:
            rect = find_largest_rect(x, y)
            # Check if any pixel within the candidate rectangle has a value of 0 in the original binary mask
            if np.sum(mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] == 0) / (rect[2] * rect[3]) > max_zero_percentage:
                continue
            # Filter out rectangles with aspect ratio above the threshold or below a minimum value
            aspect_ratio = rect[2] / rect[3]
            if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1/aspect_ratio_threshold:
                continue
            # Compute the area of the rectangle and its union with previously selected rectangles
            overlap_area = sum([rect_overlap_area(
                rect, prev_rect) for prev_rect in regions])
            if overlap_area > 0.5 * rect[2] * rect[3]:
                continue
            regions.append(rect)

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
    for x, y, w, h in unique_regions:
        cv2.rectangle(final_img, (x, y), (x + w - 1, y + h - 1), (0, 0, 0), 2)

    # Get the x, y coordinates of the top k regions and print them
    coords = [(x1, y1, x1 + w, y1 + h) for x1, y1, w, h in unique_regions]

    return coords

def get_final_img_and_mask(img):

    # # Perform object detection on the image
    # copy img to a new variable
    img_copy = img.copy()
    results = model(img_copy)
    res = results.pandas().xyxy[0]

    for obj in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = obj[:6]
        label = model.names[int(cls)]
        print(
            f'Found {label} with confidence {conf:.2f} at ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})')
        # Remove the detected objects from the image by drawing a black rectangle around them
        cv2.rectangle(img_copy, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 0, 0), -1)

    # Load the image
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # Perform adaptive thresholding to obtain a binary image
    block_size = 9
    constant = 3
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, block_size, constant)

    # Create a binary mask to initialize the GrabCut algorithm
    mask = np.zeros(img.shape[:2], np.uint8)

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

    return img_copy, mask

def inpaint_text(img, pipeline):
    # read the image
    # img = keras_ocr.tools.read(img_path)

    # Recogize text (and corresponding regions)
    # Each list of predictions in prediction_groups is a list of
    # (word, box) tuples.
    prediction_groups = pipeline.recognize([img])

    # Define the mask for inpainting
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        # For the line thickness, we will calculate the length of the line between
        # the top-left corner and the bottom-left corner.
        thickness = int(math.sqrt((x2 - x1)**3 + (y2 - y1)**3))

        # Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
                 thickness)
        # show it on the image and pause for a keypress
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return (inpainted_img)

def remove_text_from_img(img):
    # keras-ocr will automatically download pretrained
    # weights for the detector and recognizer.
    pipeline = keras_ocr.pipeline.Pipeline()
    img_text_removed = inpaint_text(img, pipeline)
    # plt.imshow(img_text_removed)
    # cv2.imshow("image", img_text_removed)
    # cv2.waitKey(0)
    return img_text_removed


# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img_path = '4.jpg'
img = cv2.imread(img_path)
no_text_img = remove_text_from_img(img)
# cv2.imshow("no text image", no_text_img)
# cv2.waitKey(0)
# save in a file
# cv2.imwrite("no_text_image.jpg", no_text_img)
# Find the top 5 largest rectangular regions not covered by the bounding boxes
final_img, mask = get_final_img_and_mask(no_text_img)

#take the mask and show it in the image
# cv2.imshow("mask", mask)
# cv2.waitKey(0)
coords = find_top_k_rectangles_new(
    no_text_img, ~mask, 5, aspect_ratio_threshold=3, max_zero_percentage=0.0)
# print coords and type of cords
res_image = add_text_to_image(
    text="get burgers hello world", raw_img=no_text_img, coords=coords[0], centered=False)


cv2.imshow("final image", res_image)
cv2.waitKey(0)

quit()

# Functions that are interesting to add:
# Bolding the words in the text that are meaningful based on the topics
# Modifying the color of the text between important and non important words
