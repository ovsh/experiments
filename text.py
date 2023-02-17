from detextify.paddle_text_detector import PaddleTextDetector
from detextify.inpainter import LocalSDInpainter
from detextify.detextifier import Detextifier

detextifier = Detextifier(PaddleTextDetector(), LocalSDInpainter())
detextifier.detextify("1.jpg", "/output/1.png")
detextifier.detextify("2.jpg", "/output/2.png")
detextifier.detextify("3.jpg", "/output/3.png")
detextifier.detextify("4.jpg", "/output/4.png")
detextifier.detextify("5.jpg", "/output/5.png")