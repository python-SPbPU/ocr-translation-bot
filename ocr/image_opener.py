import cv2 as cv

def view_image(image):
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


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

