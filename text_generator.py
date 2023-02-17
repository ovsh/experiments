# # from PIL import Image, ImageDraw, ImageFont

# # # Load the image
# # img = Image.open("image.jpeg")

# # # Create ImageDraw object
# # draw = ImageDraw.Draw(img)

# # # Select font and font size
# # font = ImageFont.truetype("arial.ttf", 36)

# # # Determine the width of the text
# # text_width = draw.textlength("Ad Headline", font=font)

# # # Determine the position of the text based on image analysis
# # text_x = (img.width - text_width) / 2
# # text_y = img.height / 4

# # # Place the text on the image
# # draw.text((text_x, text_y), "Ad Headline", fill=(255, 255, 255), font=font)

# # # Repeat the process for the subheader
# # subheader_font = ImageFont.truetype("arial.ttf", 24)
# # subheader_width = draw.textlength("Ad Subheader", font=subheader_font)
# # subheader_x = (img.width - subheader_width) / 2
# # subheader_y = text_y + 40
# # draw.text((subheader_x, subheader_y), "Ad Subheader",
# #           fill=(255, 255, 255), font=subheader_font)

# # # Save the final image
# # img.save("final_image.jpg")


# # the below code finds the biggest contours

# # import cv2
# # import numpy as np

# # # Load the image
# # img = cv2.imread("4.jpg")

# # # Convert the image to grayscale
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # Use the Canny edge detector to detect edges in the image
# # edges = cv2.Canny(gray, 1, 5)

# # # Find contours in the image
# # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # # Sort the contours by area
# # contours = sorted(contours, key=cv2.contourArea, reverse=True)

# # # Draw the 5 largest contours
# # for i in range(5):
# #     if i >= len(contours):
# #         break
# #     x, y, w, h = cv2.boundingRect(contours[i])
# #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

# # # save the final imagfe in a file
# # cv2.imwrite("final_image.jpg", img)

# # # Show the image
# # cv2.imshow("Contours", img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# #  find empty space: This code will find all the contours in the image and sort them based on their size, then exclude the 5 largest contours, which are assumed to be objects of interest. The text will then be placed in the first empty space found in the image.


# # import cv2
# # import numpy as np


# # def find_empty_space(image):
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# #     contours, _ = cv2.findContours(
# #         thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     empty_space = []
# #     for contour in contours:
# #         x, y, w, h = cv2.boundingRect(contour)
# #         empty_space.append([x, y, x+w, y+h])

# #     # sort the empty space based on its size
# #     empty_space = sorted(empty_space, key=lambda x: (
# #         x[2] - x[0]) * (x[3] - x[1]), reverse=True)

# #     # exclude the 5 largest contours
# #     empty_space = empty_space[5:]

# #     # return the empty space
# #     return empty_space


# # def place_text_in_empty_space(image, text):
# #     empty_space = find_empty_space(image)

# #     # place the text in the first empty space
# #     x1, y1, x2, y2 = empty_space[0]
# #     cv2.putText(image, text, (x1, y1),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# #     return image


# # if __name__ == '__main__':
# #     # Load the image
# #     image = cv2.imread('image.jpeg')

# #     # Place the text in the empty space
# #     image_with_text = place_text_in_empty_space(image, 'Ad Copy')

# #     # Save the image
# #     cv2.imwrite('image_with_text.jpeg', image_with_text)


# # import cv2
# # import numpy as np


# # def find_empty_space(image):
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# #     contours, _ = cv2.findContours(
# #         thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     empty_space = []
# #     for contour in contours:
# #         x, y, w, h = cv2.boundingRect(contour)
# #         empty_space.append([x, y, x+w, y+h])

# #     # sort the empty space based on its size
# #     empty_space = sorted(empty_space, key=lambda x: (
# #         x[2] - x[0]) * (x[3] - x[1]), reverse=True)

# #     # exclude the 5 largest contours
# #     empty_space = empty_space[5:]

# #     # return the empty space
# #     return empty_space


# # def draw_boxes_around_empty_space(image, color=(0, 0, 0)):
# #     empty_space = find_empty_space(image)

# #     for x1, y1, x2, y2 in empty_space:
# #         cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

# #     return image


# # # Load the image
# # image = cv2.imread('image.jpeg')

# # # Draw boxes around the empty space
# # image_with_boxes = draw_boxes_around_empty_space(image)

# # # Save the image
# # cv2.imwrite('image_with_boxes.jpeg', image_with_boxes)


# # import cv2
# # import numpy as np

# # def find_empty_space(image):
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #     _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# #     contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     empty_space = []
# #     for contour in contours:
# #         x, y, w, h = cv2.boundingRect(contour)
# #         empty_space.append([x, y, x+w, y+h])

# #     # sort the empty space based on its size
# #     empty_space = sorted(empty_space, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

# #     # only keep the 5 largest contours
# #     empty_space = empty_space[:5]

# #     # return the empty space
# #     return empty_space

# # def draw_boxes_around_empty_space(image):
# #     empty_space = find_empty_space(image)

# #     for x1, y1, x2, y2 in empty_space:
# #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)

# #     return image

# # # Load the image
# # image = cv2.imread("image.jpeg")

# # # Draw boxes around the largest empty spaces
# # image_with_boxes = draw_boxes_around_empty_space(image)

# # # Save the image
# # cv2.imwrite("image_with_boxes.jpeg", image_with_boxes)

# # import cv2
# # import numpy as np
# # import skimage.exposure

# # # read image
# # img = cv2.imread('image.jpeg')

# # # convert to gray
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # threshold to binary and invert so background is white and xxx are black
# # thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
# # thresh = 255 - thresh

# # # add black border around threshold image to avoid corner being largest distance
# # thresh2 = cv2.copyMakeBorder(thresh, 1, 1, 1, 1, cv2.BORDER_CONSTANT, (0))
# # h, w = thresh2.shape

# # # create zeros mask 2 pixels larger in each dimension
# # mask = np.zeros([h + 2, w + 2], np.uint8)

# # # apply distance transform
# # distimg = thresh2.copy()
# # distimg = cv2.distanceTransform(distimg, cv2.DIST_L2, 5)

# # # remove excess border
# # distimg = distimg[1:h-1, 1:w-1]

# # # get max value and location in distance image
# # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(distimg)

# # # scale distance image for viewing
# # distimg = skimage.exposure.rescale_intensity(
# #     distimg, in_range='image', out_range=(0, 255))
# # distimg = distimg.astype(np.uint8)

# # # draw circle on input
# # result = img.copy()
# # centx = max_loc[0]
# # centy = max_loc[1]
# # radius = int(max_val)
# # cv2.circle(result, (centx, centy), radius, (0, 0, 255), 1)
# # print('center x,y:', max_loc, 'center radius:', max_val)

# # # save image
# # cv2.imwrite('xxx_distance.png', distimg)
# # cv2.imwrite('xxx_radius.png', result)

# # # show the images
# # cv2.imshow("thresh", thresh)
# # cv2.imshow("thresh2", thresh2)
# # cv2.imshow("distance", distimg)
# # cv2.imshow("result", result)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # import cv2
# # import numpy as np

# # # Load the image
# # img = cv2.imread("4.jpg")

# # # Convert the image to grayscale
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # # Use the Canny edge detector to detect edges in the image
# # edges = cv2.Canny(gray, 1, 5)

# # # Find contours in the image
# # contours, _ = cv2.findContours(
# #     edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # # Sort the contours by area
# # contours = sorted(contours, key=cv2.contourArea, reverse=True)

# # # Draw the largest contour
# # x, y, w, h = cv2.boundingRect(contours[0])
# # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

# # # Create a mask for the object and invert it to get the background
# # mask = np.zeros(img.shape, dtype=np.uint8)
# # cv2.drawContours(mask, [contours[0]], -1, (255, 255, 255), -1)
# # mask = cv2.bitwise_not(mask)

# # # Get the sum of all pixels in the background
# # background_pixels = cv2.sumElems(mask)

# # # Get the sum of all pixels in the rectangular window
# # window = img[y:y + h, x:x + w]
# # window_pixels = cv2.sumElems(window)

# # # Calculate the average color of the background
# # background_color = background_pixels[0] // (mask.shape[0] * mask.shape[1])

# # # Calculate the average color of the rectangular window
# # window_color = window_pixels[0] // (window.shape[0] * window.shape[1])

# # # Check which color is greater and pick the rectangle in the corresponding area
# # if background_color > window_color:
# #     result = img.copy()
# #     x, y = max_loc
# #     width = int(max_val * 2)
# #     height = int(max_val * 2)
# #     cv2.rectangle(result, (x - width//2, y - height//2),
# #                   (x + width//2, y + height//2), (0, 0, 255), 1)
# #     print('center x,y:', max_loc)
# # else:
# #     result = img[y:y + h, x:x + w].copy()
# #     x = w // 2
# #     y = h // 2
# #     width = int(max_val * 2)
# #     height = int(max_val * 2)
# #     cv2.rectangle(result, (x - width//2, y - height//2),
# #                   (x + width//2, y + height//2), (0, 0, 255), 1)
# #     print('center x,y:', (x, y))

# # # Save the Save the final image
# # cv2.imwrite("final_image.jpg", result)

# # # Show the final image
# # cv2.imshow("Final Image", result)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()


# # import cv2
# # import numpy as np
# # from PIL import Image, ImageDraw, ImageFont
# # from shapely.geometry import box, Polygon

# # # Define the box intersection function


# # def box_intersection(box1, box2):
# #     poly1 = Polygon(box(box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]))
# #     poly2 = Polygon(box(box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]))
# #     intersection = poly1.intersection(poly2)
# #     return intersection.area / poly1.area


# # # Load the image
# # img = cv2.imread('4.jpg')

# # # Convert to grayscale and apply Canny edge detection
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # edges = cv2.Canny(gray, 100, 200)

# # # Find contours in the image
# # contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# # # Define a list to store the bounding boxes and scores
# # boxes = []

# # # Loop over the contours and calculate the score for each region
# # for contour in contours:
# #     x, y, w, h = cv2.boundingRect(contour)
# #     area = w * h
# #     if area > 1000:  # Only consider regions with a minimum area
# #         score = area  # Define score as area
# #         boxes.append((x, y, w, h, score))

# # # Sort the bounding boxes by score in descending order
# # boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

# # # Define the area of the unobstructed background section of the image
# # background_area = img.shape[0] * img.shape[1] - cv2.countNonZero(edges)

# # # Loop over the top 5 bounding boxes and select the one with the highest score and lowest intersection with the background
# # for i in range(5):
# #     x, y, w, h, score = boxes[i]
# #     intersection = box_intersection(
# #         (x, y, w, h), (0, 0, img.shape[1], img.shape[0]))
# #     print(intersection)
# #     if intersection < 1.5:
# #         print("Selected box:", x, y, w, h, score)
# #         # Define the position and size of the bounding box
# #         top_left = (x, y)
# #         bottom_right = (x+w, y+h)
# #         size = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])

# #         # Define the text to be displayed
# #         header_text = "Header Text"
# #         subheader_text = "Subheader Text"

# #         # Calculate the font size and padding
# #         # font = ImageFont.truetype("path/to/font.ttf", font_size)
# #         font_size = 20
# #         font =  ImageFont.load_default()
# #         # while font.getlength(header_text) < size[0] and font.getlength(header_text) < size[1]/2:
# #         #     font_size += 1
# #         header_padding = ((size[0] - font.getlength(header_text)) / 2, (size[1]/2 - font.getlength(header_text)) / 2)
# #         subheader_padding = ((size[0] - font.getlength(subheader_text)) / 2, size[1]/2 + (size[1]/2 - font.getlength(subheader_text)) / 2)

# #         # Create a new image and add the header and subheader text to it
# #         text_img = Image.new('RGB', size, color=(255, 255, 255))
# #         draw = ImageDraw.Draw(text_img)
# #         draw.text(header_padding, header_text, font=font, fill=(0, 0, 0))
# #         draw.text(subheader_padding, subheader_text, font=font, fill=(0, 0, 0))

# #         # Paste the text image onto the original image at the selected position
# #         img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# #         img_pil.paste(text_img, top_left)

# #         # Convert the image back to OpenCV format and save it to disk
# #         img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
# #         cv2.imwrite("output_image.jpg", img)

# # # Display the image for verification
# # cv2.imshow("Image with Text", img)
# # cv2.waitKey(0)


# import cv2
# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# from shapely.geometry import box, Polygon

# # Define the box intersection function


# def box_intersection(box1, box2):
#     poly1 = Polygon(box(box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]))
#     poly2 = Polygon(box(box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]))
#     intersection = poly1.intersection(poly2)
#     return intersection.area / poly1.area


# # Load the image
# img = cv2.imread('4.jpg')

# # Divide the image into a grid of cells
# cell_size = 100
# cells = [(x, y, cell_size, cell_size) for x in range(0, img.shape[1], cell_size)
#          for y in range(0, img.shape[0], cell_size)]

# # Calculate the color variance within each cell
# variances = []
# for cell in cells:
#     x, y, w, h = cell
#     patch = img[y:y+h, x:x+w]
#     variance = np.var(patch)
#     variances.append(variance)

# # Find the top cell with the lowest variance
# top_cell = cells[np.argmin(variances)]
# x, y, w, h = top_cell
# top_left = (x, y)
# bottom_right = (x+w, y+h)
# size = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])

# # Define the text to be displayed
# header_text = "Header Text"
# subheader_text = "Subheader Text"

# # Calculate the font size and padding
# font_size = 20
# font = ImageFont.load_default()
# # while font.getsize(header_text)[0] < size[0] and font.getsize(header_text)[1] < size[1]/2:
# #     font_size += 1
# #     font = ImageFont.truetype("path/to/font.ttf", font_size)
# header_padding = ((size[0] - font.getlength(header_text)) / 2,
#                   (size[1]/2 - font.getlength(header_text)) / 2)
# subheader_padding = ((size[0] - font.getlength(subheader_text)) / 2,
#                      size[1]/2 + (size[1]/2 - font.getlength(subheader_text)) / 2)

# # Create a new image and add the header and subheader text to it
# text_img = Image.new('RGB', size, color=(255, 255, 255))
# draw = ImageDraw.Draw(text_img)
# draw.text(header_padding, header_text, font=font, fill=(0, 0, 0))
# draw.text(subheader_padding, subheader_text, font=font, fill=(0, 0, 0))

# # Paste the text image onto the original image at the selected position
# img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# img_pil.paste(text_img, top_left)

# # Convert the image back to OpenCV format and display it
# img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
# cv2.imshow("Image with Text", img)
# cv2.imwrite("ff.jpg", img)
# cv2.waitKey(0)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Define the box intersection function


def box_intersection(box1, box2):
    poly1 = Polygon(box(box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]))
    poly2 = Polygon(box(box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]))
    intersection = poly1.intersection(poly2)
    return intersection.area / poly1.area


# Load the image
img = cv2.imread('4.jpg')

# Compute the variance of the color values in each pixel of the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
color_var = cv2.Laplacian(gray, cv2.CV_64F).var()

# Threshold the color variance to identify objects in the image
threshold = 100
mask = cv2.threshold(color_var, threshold, 255, cv2.THRESH_BINARY)[1]
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
_, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
mask = np.uint8(mask)

# Find contours in the mask and sort them by area
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Remove the objects from the image by drawing a black rectangle around them
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)

# Find the bounding box of the remaining image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(cnt) for cnt in contours]
x, y, w, h = max(rects, key=lambda r: r[2] * r[3])

# Calculate the size of the unobstructed area and the padding for the text
size = (w, h)
font_size = 30
font = ImageFont.load_default()
header_text = "Header Text"
subheader_text = "Subheader Text"
# while font.getsize(header_text)[0] < size[0] and font.getsize(header_text)[1] < size[1]/2:
#     font_size += 1
#     font = ImageFont.truetype("path/to/font.ttf", font_size)
header_padding = ((size[0] - font.getlength(header_text)) / 2,
                  (size[1]/2 - font.getlength(header_text)) / 2)
subheader_padding = ((size[0] - font.getlength(subheader_text)) / 2,
                     size[1]/2 + (size[1]/2 - font.getlength(subheader_text)) / 2)

# Create a new image with an alpha channel and add the header and subheader text to it
text_img = Image.new('RGBA', size, (255, 255, 255, 0))
draw = ImageDraw.Draw(text_img)
draw.text(header_padding, header_text, font=font, fill=(0, 0, 0))
draw.text(subheader_padding, subheader_text, font=font, fill=(0, 0, 0))

# Paste the text image onto the original image at the selected position
img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img_pil.paste(text_img, (x, y))

# Convert the image back to OpenCV format and display it
img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
cv2.imshow("Image with Text", img)
cv2.imwrite("ff.jpg", img)
cv2.waitKey(0)
