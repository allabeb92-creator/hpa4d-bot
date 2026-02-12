import os
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')

data_db = {"16:00": []}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 HPA-4D v4.0\n\n"
        "/input 16:00 1234 - input result 16:00\n"
        "/prediksi 19:00 - prediksi 19:00\n"
        "/status - cek database"
    )

async def input_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        _, slot, num = update.message.text.split()
        num = int(num)
        if slot == "16:00":
            data_db["16:00"].append(num)
            await update.message.reply_text(f"✅ Result {slot} = {num:04d} tersimpan")
        else:
            await update.message.reply_text("❌ Slot belum didukung")
    except:
        await update.message.reply_text("⚠️ Format: /input 16:00 1234")

async def prediksi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if data_db["16:00"]:
        mean = sum(data_db["16:00"][-5:]) // len(data_db["16:00"][-5:])
        preds = [mean % 10000, (mean+123)%10000, (mean-123)%10000, 5122, 3322]
    else:
        preds = [5122, 3322, 2424, 7090, 7416]
    msg = "🎯 PREDIKSI 19:00\n" + "\n".join([f"{i+1}. {p:04d}" for i,p in enumerate(preds[:5])])
    await update.message.reply_text(msg)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = len(data_db["16:00"])
    await update.message.reply_text(f"📊 Database: {total} angka 16:00 tersimpan")

def main():
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not token:
        print("ERROR: Token tidak ditemukan")
        return
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("input", input_result))
    app.add_handler(CommandHandler("prediksi", prediksi))
    app.add_handler(CommandHandler("status", status))
    print("✅ Bot started!")
    app.run_polling()

if __name__ == "__main__":
    main()
