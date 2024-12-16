import cv2 as cv
import easyocr

def view_image(image):
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def is_on_same_line(y1, y2, threshold=10):
    return abs(y1 - y2) < threshold


class ImageOpener:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = self.__open_image()

    def __open_image(self):
        return cv.imread(self.image_path)

    def __binarize_image(self, img, low_threshold=150, high_threshold=255):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(gray, low_threshold, high_threshold, cv.THRESH_BINARY)
        return bw

    def __enhance_contrast(self, img, clip_limit=30.0, tile_grid_size=(8, 8)):
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        bw = clahe.apply(img)
        return bw

    def process_image(self, low_threshold=150, high_threshold=255, clip_limit=30.0, tile_grid_size=(8, 8)):
        bw = self.__binarize_image(self.img, low_threshold, high_threshold)
        bw = self.__enhance_contrast(bw, clip_limit, tile_grid_size)
        return bw
    
    def write_image(self, img, path):
        cv.imwrite(path, img)


reader = easyocr.Reader(['en'])

im_o = ImageOpener("ocr/sample.png")
bw = im_o.process_image(low_threshold=150, high_threshold=255, clip_limit=30.0, tile_grid_size=(8, 8))
im_o.write_image(bw, "bw.png")

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
