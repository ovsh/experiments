import cv2
import numpy as np
import darknet

# Load the YOLO model
net = darknet.load_net("path/to/config_file.cfg",
                       "path/to/weights_file.weights", 0)
meta = darknet.load_meta("path/to/meta_file.data")

# Load the image
img = cv2.imread("path/to/image.jpg")

# Run object detection
results = darknet.detect_image(net, meta, img)

# Extract the bounding boxes of the detected objects
boxes = []
for result in results:
    label = result[0]
    confidence = result[1]
    x, y, w, h = result[2]
    left = int(x - w/2)
    top = int(y - h/2)
    right = int(x + w/2)
    bottom = int(y + h/2)
    boxes.append((left, top, right, bottom))

# Find the bounding box of the remaining image
x, y, w, h = [0, 0, img.shape[1], img.shape[0]]
for box in boxes:
    if box_intersection(box, (x, y, w, h)) > 0.5:
        x = max(x, box[0])
        y = max(y, box[1])
        w = min(w, box[2]-x)
        h = min(h, box[3]-y)

# Calculate the size of the unobstructed area and the padding for the text
size = (w, h)
font_size = 1
font = ImageFont.truetype("path/to/font.ttf", font_size)
while font.getsize(header_text)[0] < size[0] and font.getsize(header_text)[1] < size[1]/2:
    font_size += 1
    font = ImageFont.truetype("path/to/font.ttf", font_size)
header_padding = ((size[0] - font.getlength(header_text)) / 2,
                  (size[1]/2 - font.getlength(header_text)) / 2)
subheader_padding = ((size[0] - font.getlength(subheader_text)) / 2,
                     size[1]/2 + (size[1]/2 - font.getlength(subheader_text)) / 2)

# Create a new image and add the header and subheader text to it
text_img = Image.new('RGB', size, color=(255, 255, 255))
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
