from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


# Функция для создания клавиатуры выбора языка
def language_keyboard():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Русский -> Английский", callback_data="translate_ru_en"),
            InlineKeyboardButton(text="Английский -> Русский", callback_data="translate_en_ru")
        ]
    ])
    return keyboard