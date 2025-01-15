from googletrans import Translator


def translate_text_google(text: str, source_lang: str, destination_lang: str) -> str:
    translator = Translator()
    translated = translator.translate(text, src=source_lang, dest=destination_lang)
    return translated.text


# # Пример использования
# text = "Hello, World!"
# translated_text = translate_text_google(text, "en", "ru")
# print(f"Перевод: {translated_text}")
