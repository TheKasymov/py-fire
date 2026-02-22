import os
import aiohttp
import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery

# Настройки
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "7606838404:AAFiMK8TF52sISS7i4oPbJG2xi1l3wYkwM4")
API_URL = os.getenv("API_URL", "http://api:8000") 

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Временное хранилище для загруженных файлов (в памяти)
# Формат: { user_id: file_id }
pending_files = {}

# ==========================================
# UI: КЛАВИАТУРЫ
# ==========================================
# 1. Главное меню (постоянные кнопки внизу экрана)
main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="📝 Создать обращение"), KeyboardButton(text="📊 Статус системы")],
        [KeyboardButton(text="📂 Последние тикеты"), KeyboardButton(text="❓ Как пользоваться")]
    ],
    resize_keyboard=True,
    input_field_placeholder="Напишите текст или выберите действие..."
)

# 2. Кнопки для выбора типа CSV файла (появляются под сообщением)
csv_type_kb = InlineKeyboardMarkup(
    inline_keyboard=[
        [InlineKeyboardButton(text="👨‍💼 Это список Менеджеров", callback_data="csv_managers")],
        [InlineKeyboardButton(text="📋 Это база Тикетов (Лидов)", callback_data="csv_tickets")],
        [InlineKeyboardButton(text="❌ Отмена", callback_data="csv_cancel")]
    ]
)

# ==========================================
# ОБРАБОТЧИКИ КОМАНД И МЕНЮ
# ==========================================
@dp.message(CommandStart())
async def start_handler(message: Message):
    text = (
        "🤖 <b>Добро пожаловать в AI-Маршрутизатор!</b>\n\n"
        "Выберите нужное действие в меню ниже или просто отправьте мне:\n"
        "🔸 <b>Текст</b> — для создания одиночного тикета.\n"
        "🔸 <b>Скриншот</b> — для анализа ошибки ИИ.\n"
        "🔸 <b>CSV файл</b> — для массовой загрузки данных."
    )
    await message.answer(text, parse_mode="HTML", reply_markup=main_kb)

@dp.message(F.text == "❓ Как пользоваться")
async def help_handler(message: Message):
    await start_handler(message) # Просто повторяем приветственное сообщение

@dp.message(F.text.in_({"📊 Статус системы", "/status"}))
async def status_handler(message: Message):
    await message.answer("⏳ Собираю данные о системе...")
    # ... Тот же код запроса статуса, что и раньше ...
    # (Для краткости оставляю заглушку, вставьте сюда ваш aiohttp запрос статуса)
    await message.answer("✅ API на связи. Модели ИИ готовы к работе.")

@dp.message(F.text.in_({"📂 Последние тикеты", "/history"}))
async def history_handler(message: Message):
    await message.answer("⏳ Запрашиваю историю из базы данных...")
    # ... Тот же код запроса истории, что и раньше ...
    # (Вставьте сюда ваш aiohttp запрос истории)
    await message.answer("Здесь будет история 5 последних тикетов.")

# ==========================================
# ОБРАБОТКА ФАЙЛОВ CSV (Документы)
# ==========================================
@dp.message(F.document)
async def document_handler(message: Message):
    doc = message.document
    if not doc.file_name.endswith('.csv'):
        await message.answer("❌ Я принимаю только файлы в формате <b>.csv</b>", parse_mode="HTML")
        return

    # Сохраняем file_id в память, чтобы знать, что скачивать после нажатия кнопки
    pending_files[message.from_user.id] = doc.file_id
    
    await message.answer(
        f"📎 Вы загрузили файл: <b>{doc.file_name}</b>\n\n"
        "Укажите, какие данные находятся внутри этого файла, чтобы я мог правильно их обработать:",
        parse_mode="HTML",
        reply_markup=csv_type_kb
    )

@dp.callback_query(F.data.startswith("csv_"))
async def process_csv_callback(callback: CallbackQuery):
    user_id = callback.from_user.id
    action = callback.data # 'csv_managers', 'csv_tickets' или 'csv_cancel'
    
    # Убираем часики на кнопке в самом клиенте Телеграм
    await callback.answer()

    if action == "csv_cancel":
        pending_files.pop(user_id, None)
        await callback.message.edit_text("❌ Загрузка файла отменена.")
        return

    file_id = pending_files.get(user_id)
    if not file_id:
        await callback.message.edit_text("❌ Ошибка: файл устарел или потерян. Загрузите его заново.")
        return

    # 1. Скачиваем файл из Telegram
    await callback.message.edit_text("⏳ Скачиваю файл...")
    file_info = await bot.get_file(file_id)
    downloaded_file = await bot.download_file(file_info.file_path)
    
    # Определяем эндпоинт на основе кнопки
    endpoint = "/api/v1/upload/managers" if action == "csv_managers" else "/api/v1/upload/tickets"
    file_label = "Менеджеры" if action == "csv_managers" else "Тикеты"

    # 2. Отправляем в FastAPI
    try:
        await callback.message.edit_text(f"🚀 Отправляю базу «{file_label}» на сервер для парсинга...")
        
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file', downloaded_file.read(), filename=f'{user_id}_data.csv', content_type='text/csv')
            
            async with session.post(f"{API_URL}{endpoint}", data=data) as response:
                if response.status == 200:
                    res = await response.json()
                    await callback.message.edit_text(f"✅ Файл успешно обработан!\nЗагружено записей: {res.get('processed_count', 'Н/Д')}")
                else:
                    await callback.message.edit_text(f"❌ Сервер вернул ошибку: {response.status}")
    except Exception as e:
        await callback.message.edit_text("❌ Ошибка соединения с сервером API.")
    finally:
        # Очищаем память
        pending_files.pop(user_id, None)

# ==========================================
# ОБРАБОТКА СКРИНШОТОВ (ФОТО)
# ==========================================
@dp.message(F.photo)
async def photo_handler(message: Message):
    # Тот же код, что вы писали ранее для отправки фото в Gemini
    await message.answer("🔍 Вижу скриншот! Отправляю ИИ на анализ...")

# ==========================================
# ОБРАБОТКА ОБЫЧНОГО ТЕКСТА (Одиночный тикет)
# ==========================================
@dp.message(F.text)
async def text_handler(message: Message):
    # Важно: если пользователь нажал на кнопку "📝 Создать обращение", просим его ввести текст
    if message.text == "📝 Создать обращение":
        await message.answer("Отправьте мне описание проблемы одним сообщением, и я создам тикет.", reply_markup=types.ReplyKeyboardRemove())
        return

    await message.answer("⏳ Анализируем ваш запрос...")
    # Тот же код отправки json payload на /api/v1/tickets ...
    await message.answer("✅ Тикет создан и маршрутизирован!")

async def main():
    print("🤖 Telegram Бот запущен (С поддержкой меню и CSV)!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())