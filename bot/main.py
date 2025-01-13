import asyncio
from aiogram import Bot, Dispatcher
from dotenv import load_dotenv
from commands import router as commands_router

import os
load_dotenv()

async def main():
    bot = Bot(os.getenv('TOKEN'))
    dp = Dispatcher()

    dp.include_router(commands_router)

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
