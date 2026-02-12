import os
import asyncio
import asyncpg
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
DATABASE_URL = os.environ.get('DATABASE_URL')

async def init_db():
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id SERIAL PRIMARY KEY,
            slot TEXT NOT NULL,
            number INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    ''')
    await conn.close()

async def save_result(slot, number):
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute('INSERT INTO results (slot, number) VALUES ($1, $2)', slot, number)
    await conn.close()

async def get_recent_numbers(slot, limit=5):
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch('SELECT number FROM results WHERE slot = $1 ORDER BY created_at DESC LIMIT $2', slot, limit)
    await conn.close()
    return [row['number'] for row in rows]

async def get_total_count():
    conn = await asyncpg.connect(DATABASE_URL)
    count = await conn.fetchval('SELECT COUNT(*) FROM results')
    await conn.close()
    return count

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 HPA-4D v4.0 (PostgreSQL)\n\n"
        "/input 16:00 1234 - input result 16:00\n"
        "/prediksi 19:00 - prediksi 19:00\n"
        "/status - cek database"
    )

async def input_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        _, slot, num = update.message.text.split()
        num = int(num)
        if slot == "16:00":
            await save_result(slot, num)
            await update.message.reply_text(f"✅ Result {slot} = {num:04d} tersimpan di PostgreSQL")
        else:
            await update.message.reply_text("❌ Slot belum didukung")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def prediksi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        numbers = await get_recent_numbers('16:00', 5)
        if numbers:
            mean = sum(numbers) // len(numbers)
            preds = [mean % 10000, (mean+123)%10000, (mean-123)%10000, 5122, 3322]
        else:
            preds = [5122, 3322, 2424, 7090, 7416]
        msg = "🎯 PREDIKSI 19:00\n" + "\n".join([f"{i+1}. {p:04d}" for i,p in enumerate(preds[:5])])
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = await get_total_count()
    await update.message.reply_text(f"📊 Database PostgreSQL: {total} angka tersimpan")

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(init_db())
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("input", input_result))
    app.add_handler(CommandHandler("prediksi", prediksi))
    app.add_handler(CommandHandler("status", status))
    print("✅ Bot started with PostgreSQL!")
    app.run_polling()

if __name__ == "__main__":
    main()
