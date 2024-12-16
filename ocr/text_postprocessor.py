from typing import Dict, List, Tuple, Pattern
import re


class TextPostprocessor:
    def __init__(self) -> None:
        # Словарь замен для типичных ошибок распознавания
        self.replacements: Dict[str, str] = {
            '_': '.',  # точки часто распознаются как подчеркивания
            '،': ',',  # разные виды запятых
            '`': '.',  # обратные кавычки как точки
            '\'': '.',  # одиночные кавычки как точки
            '。': '.',  # азиатские точки
            '·': '.',  # центрированные точки
        }
        
        # Паттерны для исправления
        self.patterns: List[Tuple[Pattern, str]] = [
            (re.compile(r'_{2,}'), '.'),  # несколько подчеркиваний подряд заменяем на точку
            (re.compile(r'\s*_\s*'), '.'),  # подчеркивание с пробелами вокруг на точку
            (re.compile(r'\.{2,}'), '.'),  # несколько точек подряд на одну точку
            (re.compile(r'\s+\.'), '.'),  # убираем пробел перед точкой
            (re.compile(r'\.\s+'), '. '),  # нормализуем пробел после точки
        ]

    def process_text(self, text: str) -> str:
        """
        Применяет правила пост-обработки к тексту
        
        Args:
            text: Исходный текст
            
        Returns:
            Обработанный текст
        """
        # Применяем простые замены
        for old, new in self.replacements.items():
            text = text.replace(old, new)
            
        # Применяем регулярные выражения
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
            
        return text.strip() 