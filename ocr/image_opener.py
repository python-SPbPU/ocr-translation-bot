import cv2 as cv
import numpy as np
from typing import Tuple

class ImageOpener:
    def __init__(self, image_path: str) -> None:
        self.image_path: str = image_path
        self.img: np.ndarray = self.__open_image()

    def __open_image(self) -> np.ndarray:
        return cv.imread(self.image_path)

    def __binarize_image(self, img: np.ndarray, low_threshold: int = 150, high_threshold: int = 255) -> np.ndarray:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, bw = cv.threshold(gray, low_threshold, high_threshold, cv.THRESH_BINARY)
        return bw

    def __enhance_contrast(self, img: np.ndarray, clip_limit: float = 30.0, 
                          tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        bw = clahe.apply(img)
        return bw

    def process_image(self, low_threshold: int = 150, high_threshold: int = 255, 
                     clip_limit: float = 30.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        bw = self.__binarize_image(self.img, low_threshold, high_threshold)
        bw = self.__enhance_contrast(bw, clip_limit, tile_grid_size)
        return bw
    
    def write_image(self, img: np.ndarray, path: str) -> None:
        cv.imwrite(path, img)

