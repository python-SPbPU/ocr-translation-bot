import easyocr
from image_opener import ImageOpener
from text_postprocessor import TextPostprocessor
from typing import List, Tuple
import numpy as np

class TextRecognizer:
    """
    Класс для распознавания текста на изображении с предварительной обработкой
    и пост-обработкой результатов.

    Attributes:
        opener (ImageOpener): Объект для открытия и обработки изображения
        processed_image (np.ndarray): Обработанное изображение
        postprocessor (TextPostprocessor): Объект для пост-обработки распознанного текста

    Example:
        >>> recognizer = TextRecognizer("path/to/image.png")
        >>> strings = recognizer.recognize_text(langs=['ru', 'en'])
        >>> for _, text, prob in strings:
        ...     print(f"{text} (вероятность: {prob:.2%})")
    """

    def __init__(self, image_path: str, low_threshold: int = 150, high_threshold: int = 255, 
                 clip_limit: float = 30.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
        """
        Args:
            image_path: Путь к изображению
            low_threshold: Нижний порог для бинаризации
            high_threshold: Верхний порог для бинаризации
            clip_limit: Предел контраста для CLAHE
            tile_grid_size: Размер сетки для CLAHE
        """
        self.opener = ImageOpener(image_path)
        self.processed_image = self.opener.process_image(low_threshold, high_threshold, clip_limit, tile_grid_size)
        self.opener.write_image(self.processed_image, "processed_image.png")
        self.postprocessor = TextPostprocessor()
        
    def __is_on_same_line(self, y1: float, y2: float, threshold: int = 10) -> bool:
        return abs(y1 - y2) < threshold

    def recognize_text(self, langs: List[str] = ['en']) -> List[Tuple[float, str, float]]:
        """
        Распознает текст на обработанном изображении.

        Args:
            langs: Список языков для распознавания (например, ['ru', 'en'])

        Returns:
            Список кортежей (y_координата, текст, вероятность), где:
            - y_координата: позиция строки по вертикали
            - текст: распознанный текст после пост-обработки
            - вероятность: уверенность в правильности распознавания (0-1)

        Example:
            >>> strings = recognizer.recognize_text(['ru', 'en'])
            >>> for y, text, prob in strings:
            ...     print(f"Y: {y}, Текст: {text}, Вероятность: {prob:.2%}")
        """
        reader = easyocr.Reader(langs)
        results = reader.readtext(self.processed_image)
        strings = []
        i = 0
        for bbox, text, prob in results:
            processed_text = self.postprocessor.process_text(text)
            
            tl, tr, br, bl = bbox
            if len(strings) == 0:
                strings.append((tl[1], processed_text, prob))
                i += 1
            else:
                if self.__is_on_same_line(tl[1], strings[i - 1][0]):
                    prev_prob = strings[i - 1][2]
                    prev_text_len = len(strings[i - 1][1])
                    curr_text_len = len(processed_text)
                    avg_prob = (prev_prob * prev_text_len + prob * curr_text_len) / (prev_text_len + curr_text_len)
                    
                    combined_text = strings[i - 1][1] + " " + processed_text
                    processed_combined_text = self.postprocessor.process_text(combined_text)
                    
                    strings[i - 1] = (strings[i - 1][0], processed_combined_text, avg_prob)
                else:
                    strings.append((tl[1], processed_text, prob))
                    i += 1
        return strings


rec = TextRecognizer(
    "ocr/sample.png",
    low_threshold=150,
    high_threshold=255,
    clip_limit=20.0,
    tile_grid_size=(8, 8)
)
strings = rec.recognize_text(langs=['ru', 'en'])

for string in strings:
    print(string[1])

