from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command
from dotenv import load_dotenv
import aiohttp
from ocr.recognition import TextRecognizer
import os

router = Router()

load_dotenv()

# Обработчик для команды /start
@router.message(Command("start"))
async def start_command(message: Message):
    await message.answer("Привет! Я бот переводчик картинок. Отправьте комманду или фото, чтобы проверить мой функционал.")

# Обработчик для команды /help
@router.message(Command("help"))
async def help_command(message: Message):
    await message.answer("Я умею:\n1. Обрабатывать изображения.\n2. Выбирать язык перевода.")

#Обработчик для получения изображения и распознавания текста.
@router.message()
async def handle_image(message: Message):
    photo = message.photo[-1]

    # Получаем объект файла
    file_info = await message.bot.get_file(photo.file_id)

    # Загружаем файл
    file_path = file_info.file_path
    file_name = f"images/{photo.file_id}.jpg"

    # Скачиваем файл
    file_url = f"https://api.telegram.org/file/bot{os.getenv('TOKEN')}/{file_path}"
    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            if response.status == 200:
                with open(file_name, "wb") as f:
                    f.write(await response.read())
                await message.reply("Файл успешно загружен. Начинаю распознавание текста...")
                try:
                    recognizer = TextRecognizer(file_name)
                    results = recognizer.recognize_text(langs=['ru', 'en'])  # Указываем нужные языки

                    if results:
                        recognized_text = "\n".join([f"{text} (уверенность: {prob:.2%})" for _, text, prob in results])
                        await message.reply(f"Распознанный текст:\n{recognized_text}")
                    else:
                        await message.reply("Не удалось распознать текст на изображении.")
                except Exception as e:
                    await message.reply(f"Произошла ошибка при распознавании: {e}")

            else:
                await message.reply("Не удалось загрузить файл.")

