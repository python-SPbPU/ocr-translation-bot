import cv2 as cv
import numpy as np
from typing import Tuple

class ImageOpener:
    """
    Класс для открытия и предварительной обработки изображений.
    
    Выполняет:
    - Бинаризацию изображения
    - Улучшение контраста с помощью CLAHE
    
    Attributes:
        image_path (str): Путь к изображению
        img (np.ndarray): Загруженное изображение

    Example:
        >>> opener = ImageOpener("path/to/image.png")
        >>> processed = opener.process_image(low_threshold=150, high_threshold=255)
        >>> opener.write_image(processed, "output.png")
    """

    def __init__(self, image_path: str) -> None:
        """
        Args:
            image_path: Путь к изображению
        """
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
        """
        Обрабатывает изображение для улучшения качества распознавания текста.
        
        Процесс включает:
        1. Бинаризацию изображения с заданными порогами
        2. Улучшение контраста методом CLAHE
        
        Args:
            low_threshold: Нижний порог для бинаризации (0-255)
            high_threshold: Верхний порог для бинаризации (0-255)
            clip_limit: Ограничение контраста для CLAHE
            tile_grid_size: Размер сетки для CLAHE в формате (строки, столбцы)
        
        Returns:
            Обработанное изображение в формате numpy array
        
        Example:
            >>> processed = opener.process_image(
            ...     low_threshold=150,
            ...     high_threshold=255,
            ...     clip_limit=30.0,
            ...     tile_grid_size=(8, 8)
            ... )
        """
        bw = self.__binarize_image(self.img, low_threshold, high_threshold)
        bw = self.__enhance_contrast(bw, clip_limit, tile_grid_size)
        return bw
    
    def write_image(self, img: np.ndarray, path: str) -> None:
        """
        Сохраняет изображение в файл.
        
        Args:
            img: Изображение в формате numpy array
            path: Путь для сохранения файла
        
        Example:
            >>> opener.write_image(processed_image, "output.png")
        """
        cv.imwrite(path, img)

