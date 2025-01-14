import asyncio
from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand
from dotenv import load_dotenv
from commands import router as commands_router
import os


load_dotenv()

async def setup_bot_commands(bot: Bot):
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="help", description="Получить помощь"),
    ]
    await bot.set_my_commands(commands)

async def main():
    bot = Bot(os.getenv('TOKEN'), html=True)
    dp = Dispatcher()

    dp.include_router(commands_router)

    await setup_bot_commands(bot)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
