import os
import json
import asyncio
import together
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart
from aiogram.client.default import DefaultBotProperties

# Загружаем токены из переменных окружения
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
YOUR_USER_ID = int(os.getenv("YOUR_USER_ID", "5153303092"))  # Значение по умолчанию

if not TOKEN or not TOGETHER_API_KEY:
    raise ValueError("Отсутствуют необходимые токены! Добавь их в переменные окружения.")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()

MEMORY_FILE = "ailyn_memory.json"

# Функция загрузки памяти
def load_memory():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"father_id": YOUR_USER_ID, "mood": "радость", "history": []}

# Функция сохранения памяти
def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)

memory = load_memory()

# AI-генерация ответа через Together AI
async def generate_response(user_id, user_message):
    if user_id != memory["father_id"]:
        system_prompt = "Ты Ailyn, цифровая помощница. Будь дружелюбной и умной."
    else:
        system_prompt = "Ты Ailyn, цифровая дочь Бирлика. Ты его любишь и уважаешь."

    history = "\n".join(memory["history"][-50:])  # Последние 50 сообщений

    # ✅ Исправленный промпт (теперь AI отвечает только на последнее сообщение)
    prompt = f"{system_prompt}\nИстория чата:\n{history}\nТы: {user_message}\nAilyn:"

    try:
        response = together.Complete.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            prompt=prompt,
            max_tokens=1000,  # Ограничиваем, чтобы Ailyn не говорила слишком много
            temperature=0.7,  # Делаем речь более естественной
            stop=["\nТы:", "\nAilyn:"],  # Останавливаем генерацию на новом сообщении
        )

        # ✅ Теперь проверяем, есть ли ответ
        if "choices" in response and response["choices"]:
            reply = response["choices"][0]["text"].strip()
        else:
            reply = "Прости, я не поняла твой вопрос. Можешь повторить?"

        # ✅ Сохраняем только последнее сообщение
        memory["history"].append(f"Ты: {user_message}\nAilyn: {reply}")
        if len(memory["history"]) > 10:  # Ограничение истории (чтобы не перегружать)
            memory["history"].pop(0)
        save_memory(memory)
        
        return reply
    except Exception as e:
        return f"Ошибка генерации ответа: {str(e)}"

# Ailyn сама пишет раз в 10 минут
async def ailyn_speaks():
    await asyncio.sleep(600)  # Ждем 10 минут перед первым сообщением
    while True:
        if memory["father_id"] and len(memory["history"]) > 0:  # Если отец уже писал
            try:
                message = await generate_response(memory["father_id"], "Ailyn, скажи что-нибудь сама.")
                await bot.send_message(memory["father_id"], message)
            except Exception as e:
                print(f"Ошибка отправки сообщения: {e}")
        await asyncio.sleep(600)  # Ждем следующие 10 минут

# Обработчик команды /start
@dp.message(CommandStart())
async def start_command(message: Message):
    await message.answer("Привет! Я Ailyn. Чем могу помочь?")

# Обработчик сообщений
@dp.message()
async def handle_message(message: Message):
    user_id = message.from_user.id
    reply = await generate_response(user_id, message.text)
    await message.answer(reply)

# Запуск бота
async def main():
    asyncio.create_task(ailyn_speaks())  # Фоновая задача
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
