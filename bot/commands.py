from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command

router = Router()

# Обработчик для команды /start
@router.message(Command("start"))
async def start_command(message: Message):
    await message.answer("Привет! Я бот переводчик картинок. Отправьте комманду или фото, чтобы проверить мой функционал.")

# Обработчик для команды /help
@router.message(Command("help"))
async def help_command(message: Message):
    await message.answer("Я умею:\n1. Обрабатывать изображения.\n2. Выбирать язык перевода.")
