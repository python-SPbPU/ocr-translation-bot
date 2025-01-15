from aiogram import Router, types
from aiogram.types import Message
from aiogram.filters import Command
from dotenv import load_dotenv
import aiohttp
from ocr.recognition import TextRecognizer
from keyboard import language_keyboard
import os
from translation import translate_text_google


router = Router()

load_dotenv()

FILE_NAME = "inp.jpg"


# Обработчик для команды /start
@router.message(Command("start"))
async def start_command(message: Message):
    await message.answer(
        "Привет! Я бот переводчик картинок. Отправьте команду или фото, чтобы проверить мой функционал.")


# Обработчик для команды /help
@router.message(Command("help"))
async def help_command(message: Message):
    await message.answer("Я умею:\n1. Обрабатывать изображения.\n2. Выбирать язык перевода.")


# Обработчик для получения изображения и распознавания текста.
@router.message()
async def handle_image(message: Message):
    if not message.photo:
        await message.reply("Пожалуйста, отправьте изображение.")
        return

    photo = message.photo[-1]

    # Получаем объект файла
    file_info = await message.bot.get_file(photo.file_id)

    # Загружаем файл
    file_path = file_info.file_path

    # Скачиваем файл
    file_url = f"https://api.telegram.org/file/bot{os.getenv('TOKEN')}/{file_path}"
    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            if response.status == 200:
                with open(FILE_NAME, "wb") as f:
                    f.write(await response.read())
                await message.reply("Файл успешно загружен. Выберите язык перевода:", reply_markup=language_keyboard())
            else:
                await message.reply("Не удалось загрузить файл.")



# Обработчик для выбора языка перевода
@router.callback_query()
async def handle_language_selection(callback_query: types.CallbackQuery):
    callback_data = callback_query.data
    # recognized_text = None
    try:
        recognizer = TextRecognizer(FILE_NAME)
        results = recognizer.recognize_text(langs=['ru', 'en'])

        if results:
            recognized_text = "\n".join([text if prob > 0.6 else "" for _, text, prob in results])
            #await callback_query.message.reply(f"Распознанный текст:\n{recognized_text}")
        else:
            await callback_query.message.reply("Не удалось распознать текст на изображении.")
            return
    except Exception as e:
        await callback_query.message.reply(f"Произошла ошибка при распознавании: {e}")
        return
    if callback_data == "translate_ru_en":
        await callback_query.message.reply("Вы выбрали перевод с русского на английский. Начинаю обработку...")
        await callback_query.message.reply(translate_text_google(recognized_text, 'ru', 'en'))
    elif callback_data == "translate_en_ru":
        await callback_query.message.reply("Вы выбрали перевод с английского на русский. Начинаю обработку...")
        await callback_query.message.reply(translate_text_google(recognized_text, 'en', 'ru'))
    else:
        await callback_query.message.reply("Неизвестный выбор. Пожалуйста, попробуйте снова.")
