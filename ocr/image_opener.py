import cv2 as cv
import easyocr

def view_image(image):
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


reader = easyocr.Reader(['en'])

image = cv.imread("ocr/sample.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Бинаризация (преобразование в черно-белое)
_, bw = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)

# Улучшение контраста
clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(10, 10))
bw = clahe.apply(bw)

# view_image(image)
# view_image(bw)

results = reader.readtext(bw)
for bbox, text, prob in results:
    print(f"Bounding Box: {bbox}, Text: {text}, Probability: {prob}")
