import easyocr
from image_opener import ImageOpener
from text_postprocessor import TextPostprocessor
from typing import List, Tuple
import numpy as np

class TextRecognizer:
    def __init__(self, image_path: str, low_threshold: int = 150, high_threshold: int = 255, 
                 clip_limit: float = 30.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
        self.opener = ImageOpener(image_path)
        self.processed_image = self.opener.process_image(low_threshold, high_threshold, clip_limit, tile_grid_size)
        self.opener.write_image(self.processed_image, "processed_image.png")
        self.postprocessor = TextPostprocessor()
        
    def __is_on_same_line(self, y1: float, y2: float, threshold: int = 10) -> bool:
        return abs(y1 - y2) < threshold

    def recognize_text(self, langs: List[str] = ['en']) -> List[Tuple[float, str, float]]:
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
strings = rec.recognize_text()

for string in strings:
    print(string[1])

