# Optimizations TODO
# Remove print statements: The print statements are not necessary in production and can slow down the performance of the function. They should be removed.

# Improve code structure: The code structure could be improved by breaking the function down into smaller, more manageable functions. For example, the code for splitting the text into lines could be put into a separate function.

# Improve variable names: The variable names used in the function could be improved to make the code more readable and easier to understand. For example, the variable name "lines" could be changed to "text_lines".

# Add error handling: The function should include error handling to handle any unexpected errors that might occur.

# Improve performance: The performance of the function could be improved by using more efficient algorithms and data structures. For example, the loop that splits the text into lines could be optimized using a binary search algorithm.

# Use a different font library: The current font library used in the function is "arial.ttf", which might not be available on all systems. It might be better to use a font library that is more widely available, such as "Pillow".

# Add documentation: The function should include proper documentation, including a description of the function's purpose, its input parameters, its output, and any exceptions it might raise.

import matplotlib.pyplot as plt
import math
import keras_ocr
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from shapely.geometry import Polygon


class ImageTextDetector:
    """Initializes the ImageTextDetector object.

    The constructor creates a new instance of the keras_ocr.pipeline.Pipeline class, which is used to recognize text in images.
    """

    def __init__(self):
        self.pipeline = keras_ocr.pipeline.Pipeline()

    def detect(self, img):
        """Detects text in an image.

        Args:
            img: A NumPy array representing an image.

        Returns:
            A list of predictions, where each prediction is a tuple containing a bounding box and the recognized text.
            The bounding box is a list of four (x, y) coordinates representing the top-left, top-right, bottom-right, and bottom-left corners.
            The recognized text is a string.
        """
        try:
            # Get the predictions
            predictions = self.pipeline.recognize([img])

            # Get the bounding boxes
            boxes = [prediction[0] for prediction in predictions]

            # Get the text
            text = [prediction[1] for prediction in predictions]

            # Return the predictions
            return predictions
        except Exception as e:
            # Handle the exception
            print(f"Error detecting text: {e}")
            return []

    def paint_over_text(self, img, predictions):
        """Paints over the text in an image.

        Args:
            img: A NumPy array representing an image.
            predictions: A list of predictions, where each prediction is a tuple containing a bounding box and the recognized text.
            The bounding box is a list of four (x, y) coordinates representing the top-left, top-right, bottom-right, and bottom-left corners.
            The recognized text is a string.

        Returns:
            A NumPy array representing the modified image with the text painted over.
        """
        # Define the mask for inpainting
        mask = np.zeros(img.shape[:2], dtype="uint8")
        for box in predictions[0]:
            x0, y0 = box[1][0]
            x1, y1 = box[1][1]
            x2, y2 = box[1][2]
            x3, y3 = box[1][3]

            x_mid0, y_mid0 = ImageProcessor.midpoint(x1, y1, x2, y2)
            x_mid1, y_mi1 = ImageProcessor.midpoint(x0, y0, x3, y3)

            # For the line thickness, we will calculate the length of the line between
            # the top-left corner and the bottom-left corner.
            thickness = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

            # Define the line and inpaint
            cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,
                     thickness)
            # show it on the image and pause for a keypress
            inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

        return (inpainted_img)

    def remove_text_from_image(self, img):
        """Removes the text from an image.

        Args:
            img: A NumPy array representing an image.

        Returns:
            A NumPy array representing the modified image with the text removed.
        """
        # Get the predictions
        predictions = self.detect(img)

        # Paint over the text
        img = self.paint_over_text(img, predictions)

        return img


class ImageProcessor:

    # load YOLO5 model in init
    def __init__(self, debug=False):
        self.model = model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.debug = debug

    @staticmethod
    def midpoint(x1, y1, x2, y2):
        x_mid = int((x1 + x2)/2)
        y_mid = int((y1 + y2)/2)
        return (x_mid, y_mid)

    @staticmethod
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

    def add_text_to_image(self, text, raw_img, coords, padding=0.98, centered=True, split_lines=False):
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
            font_multi_line = ImageFont.truetype(
                "arial.ttf", font_size_multi_line)

            # Split the text into two lines that fit within the bounding box
            words = text.split()
            if len(words) <= 1:
                lines = [text]
            else:
                lines = []
                line = ""
                for word in words:
                    if font_multi_line.getlength(line + word) < padding * width and font_multi_line.getlength(line + word + " ") < padding * width:
                        line += word + " "
                    else:
                        lines.append(line[:-1])
                        line = word + " "
                        if len(lines) == 1:
                            max_chars_per_line = len(line)//2
                lines.append(line[:-1])

                if len(lines) == 2:
                    # Check if the second line exceeds the width of the bounding box
                    if font_multi_line.getlength(lines[1]) > padding * width:
                        # If so, try splitting the second line at the midpoint
                        midpoint = len(lines[1])//2
                        lines[1] = lines[1][:midpoint].strip()
                        lines.insert(2, lines[1][midpoint:].strip())
                    elif font_multi_line.getlength(lines[1]) + font_multi_line.getlength(lines[0]) > padding * width:
                        # If the two lines combined exceed the width of the bounding box, split the first line at the midpoint
                        midpoint = max_chars_per_line
                        lines[0] = lines[0][:midpoint].strip()
                        lines.insert(1, lines[0][midpoint:].strip())

                font_size_multi_line = font_size
                font_multi_line = ImageFont.truetype(
                    "arial.ttf", font_size_multi_line)

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
            if self.debug == True:
                print("Text drawn at ({}, {}). Text = {}".format(x, y, line))
            y += font_size

        # Print where you drew the text, including text font size and width and height
        if self.debug == True:
            print("Text drawn at ({}, {})".format(x, y))
            print("Text font size: {}".format(font_size))
            print("Text width: {}".format(font.getbbox(
                lines[0])[2] - font.getbbox(lines[0])[0]))

        # Convert the modified image back to the OpenCV format
        modified_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # draw a bounding box around the original coords in black and then draw another bounding box around the bounding box of the text in red
        if self.debug == True:
            cv2.rectangle(modified_image, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.rectangle(modified_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        return modified_image

    def remove_small_components(self, mask, min_size):
        # Perform connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        # Remove small components
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                labels[labels == i] = 0

        # Apply the new labels to the mask and remove small holes
        mask = (labels > 0).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        return mask

    def find_top_k_rectangles_new(self, img, mask, k, max_zero_percentage=0.0, aspect_ratio_threshold=4):
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
            # if self.debug == True:
            # print('best_rect: {}'.format(best_rect))
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
                subarray = mask[rect[1]:rect[1] +
                                rect[3], rect[0]:rect[0]+rect[2]]
                if np.any(subarray == 0):
                    zero_percentage = np.mean(subarray == 0)
                    # # if coordinates are 0,0,600,600 - print the zero percentage and the max zero percentage
                    # if rect[0] == 0 and rect[1] == 0 and rect[2] == 600 and rect[3] == 600:
                    #     if self.debug == True:
                    #         print('zero_percentage: {}'.format(zero_percentage))
                    #         print('max_zero_percentage: {}'.format(
                    #             max_zero_percentage))
                    if zero_percentage > max_zero_percentage:
                        continue
                # Filter out rectangles with aspect ratio above the threshold or below a minimum value
                aspect_ratio = rect[2] / rect[3]
                if aspect_ratio > aspect_ratio_threshold or aspect_ratio < 1/aspect_ratio_threshold:
                    continue
                # Compute the area of the rectangle and its union with previously selected rectangles
                overlap_area = sum([ImageProcessor.rect_overlap_area(
                    rect, prev_rect) for prev_rect in regions])
                if overlap_area > 0.5 * rect[2] * rect[3]:
                    continue
                regions.append(rect)

        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)

        # Sort the regions by area in descending order
        regions = sorted(
            regions, key=lambda region: region[2] * region[3], reverse=True)

        # Keep the top k regions
        unique_regions = []
        for region in regions:
            if not any(ImageProcessor.rect_overlap_area(region, prev_rect) >= 0.5 * min(region[2] * region[3], prev_rect[2] * prev_rect[3]) for prev_rect in unique_regions):
                unique_regions.append(region)
            if len(unique_regions) == k:
                break

        if (self.debug == True):
            print("Number of regions found: {}".format(len(unique_regions)))
            print("Regions: {}".format(unique_regions))
        # Draw rectangles around the top k regions
            for x, y, w, h in unique_regions:
                cv2.rectangle(img, (x, y),
                              (x + w - 1, y + h - 1), (0, 0, 0), 2)
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)

        # Get the x, y coordinates of the top k regions and print them
        coords = [(x1, y1, x1 + w, y1 + h) for x1, y1, w, h in unique_regions]

        return coords

    def get_final_img_and_mask(self, img):

        # # Perform object detection on the image
        # copy img to a new variable
        img_copy = img.copy()
        results = self.model(img_copy)
        res = results.pandas().xyxy[0]

        for obj in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = obj[:6]
            label = self.model.names[int(cls)]
            if self.debug == True:
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
        if self.debug == True:
            print("number of contours: ", len(contours))

        # Find the bounding boxes of the top 5 largest contours in the image
        bounding_boxes = []
        for c in sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:5]:
            x, y, w, h = cv2.boundingRect(c)
            bounding_boxes.append((x, y, w, h))

        # Draw the top 5 contours on the original image
        # These are the objects we need to remove from the image
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            if (self.debug == True):
                print(x, y, w, h)
                print("size of each contur in image: ", cv2.contourArea(c))
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, f"#{i+1}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Create a mask that marks the areas covered by the bounding boxes
        mask = np.zeros_like(gray)
        for x, y, w, h in bounding_boxes:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # remove small objects from the binary image
        # set minsize for at least 10% of the image size
        min_size = int(0.1 * img.shape[0] * img.shape[1])
        mask = self.remove_small_components(mask, min_size)
        
        return img_copy, mask

    def write_ad_copy(self, text, img, max_zero_percentage=0.0, aspect_ratio_threshold=3, split_lines=False):
        # first step is we find the rectangles in the image
        # then for top rectangle, we call add_text_to_image

        mask = self.get_final_img_and_mask(img)[1]
        # find the top 5 rectangles
        coords = self.find_top_k_rectangles_new(
            img, ~mask, k=5, max_zero_percentage=max_zero_percentage, aspect_ratio_threshold=aspect_ratio_threshold)
        # get mask to apply
        img = self.add_text_to_image(text, img, coords[0])

        return img


if __name__ == "__main__":
    img_path = '4.jpg'
    img = cv2.imread(img_path)
    # remove text from image
    ImageTextDetector = ImageTextDetector()
    no_text_img = ImageTextDetector.remove_text_from_image(img)
    ImageProcessor = ImageProcessor(debug=False)
    # write copy
    final_img = ImageProcessor.write_ad_copy(
        "buy big car", no_text_img, max_zero_percentage=0.0, aspect_ratio_threshold=3, split_lines=False)

    # show the image and wait for a keypress
    cv2.imshow("final image", final_img)
    cv2.waitKey(0)
