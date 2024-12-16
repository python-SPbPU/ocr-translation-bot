import cv2 as cv
import easyocr

def view_image(image):
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def is_on_same_line(y1, y2, threshold=10):
    return abs(y1 - y2) < threshold


reader = easyocr.Reader(['en'])

image = cv.imread("ocr/sample.png")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Бинаризация (преобразование в черно-белое)
_, bw = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)

# Улучшение контраста
clahe = cv.createCLAHE(clipLimit=30.0, tileGridSize=(8, 8))
bw = clahe.apply(bw)

cv.imwrite("bw.png", bw)

# view_image(image)
# view_image(bw)

results = reader.readtext(bw)
strings = []
i = 0
for bbox, text, prob in results:
    tl, tr, br, bl = bbox
    # print(tl, text)
    if len(strings) == 0:
        strings.append((tl, text))
        i += 1
    else:
        if is_on_same_line(tl[1], strings[i - 1][0][1]):
            strings[i - 1] = (strings[i - 1][0], strings[i - 1][1] + " " + text)
        else:
            strings.append((tl, text))
            i += 1

for string in strings:
    print(string[1])
