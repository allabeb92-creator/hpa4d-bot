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

# ========== PREDIKSI SEMUA SLOT ==========
async def prediksi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        slot = context.args[0]
        
        if slot == "13:00":
            # Prediksi 13:00 berdasarkan 00:01
            last_0001 = await get_recent_numbers('00:01', 1)
            if last_0001 and last_0001[0] < 2000:
                preds = [5602, 5740, 5923, 6977, 7822, 8042, 8958, 4283, 3300, 6000]
            else:
                preds = [4937, 5000, 4800, 5100, 4500, 5200, 4700, 5300, 4400, 5400]
                
        elif slot == "16:00":
            # Prediksi 16:00 berdasarkan 13:00
            last_1300 = await get_recent_numbers('13:00', 1)
            if last_1300:
                mean_diff = -273  # Rata-rata selisih 16:00 - 13:00
                base = last_1300[0] + mean_diff
                preds = [base % 10000, (base+123)%10000, (base-123)%10000, 4560, 4557, 4440, 4630, 4538, 4691, 4906]
            else:
                preds = [4560, 4557, 4440, 4630, 4538, 4691, 4906, 4057, 4360, 5169]
                
        elif slot == "19:00":
            # Prediksi 19:00 berdasarkan 16:00 (sudah ada)
            numbers = await get_recent_numbers('16:00', 5)
            if numbers:
                mean = sum(numbers) // len(numbers)
                preds = [mean % 10000, (mean+123)%10000, (mean-123)%10000, 5122, 3322, 2424, 7090, 7416, 1650, 4181]
            else:
                preds = [5122, 3322, 2424, 7090, 7416, 1650, 4181, 1895, 9922, 7022]
                
        elif slot == "22:00":
            # Prediksi 22:00 berdasarkan 19:00
            last_1900 = await get_recent_numbers('19:00', 1)
            if last_1900:
                mean_diff = -792  # Rata-rata selisih 22:00 - 19:00
                base = last_1900[0] + mean_diff
                preds = [base % 10000, 4613, 4500, 5392, 5479, 4495, 9063, 8841, 5098, 4624]
            else:
                preds = [4613, 4500, 5392, 5479, 4495, 9063, 8841, 5098, 4624, 5943]
                
        elif slot == "23:00":
            # Prediksi 23:00 berdasarkan 22:00 + hot number 9037
            last_2200 = await get_recent_numbers('22:00', 1)
            if last_2200:
                mean_diff = 506  # Rata-rata selisih 23:00 - 22:00
                base = last_2200[0] + mean_diff
                preds = [base % 10000, 9037, 1572, 1007, 5170, 5129, 5296, 5907, 5728, 5406]
            else:
                preds = [9037, 1572, 1007, 5170, 5129, 5296, 5907, 5728, 5406, 2059]
                
        elif slot == "00:01":
            # Prediksi 00:01 berdasarkan 23:00
            last_2300 = await get_recent_numbers('23:00', 1)
            if last_2300:
                mean_diff = -132  # Rata-rata selisih 00:01 - 23:00
                base = last_2300[0] + mean_diff
                preds = [base % 10000, 5028, 5000, 5066, 5135, 5211, 5404, 5590, 5080, 5170]
            else:
                preds = [5028, 5000, 5066, 5135, 5211, 5404, 5590, 5080, 5170, 5260]
        else:
            await update.message.reply_text("❌ Slot tidak valid. Gunakan: 13:00, 16:00, 19:00, 22:00, 23:00, 00:01")
            return
        
        msg = f"🎯 PREDIKSI {slot}\n" + "\n".join([f"{i+1}. {p:04d}" for i,p in enumerate(preds[:10])])
        await update.message.reply_text(msg)
        
    except (IndexError, ValueError):
        await update.message.reply_text("⚠️ Gunakan: /prediksi 19:00")

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
