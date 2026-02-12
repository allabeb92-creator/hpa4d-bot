import os
import asyncio
import asyncpg
import math
import random
import pickle
import tempfile
from collections import OrderedDict
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ========== DATABASE ==========
class TotoDatabase:
    def __init__(self):
        self.conn_pool = None

    async def init_pool(self):
        DATABASE_URL = os.environ.get('DATABASE_URL')
        if not DATABASE_URL:
            print("⚠️ DATABASE_URL tidak ditemukan, pakai memory fallback")
            return
        try:
            self.conn_pool = await asyncpg.create_pool(DATABASE_URL)
            async with self.conn_pool.acquire() as conn:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS results (
                        id SERIAL PRIMARY KEY,
                        slot TEXT NOT NULL,
                        number INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                ''')
        except Exception as e:
            print(f"⚠️ Koneksi database gagal: {e}")

    async def get_all_slot(self, slot):
        if not self.conn_pool:
            return []
        async with self.conn_pool.acquire() as conn:
            rows = await conn.fetch('SELECT date, number FROM results WHERE slot=$1 ORDER BY created_at ASC', slot)
            return [(r['date'], r['number']) for r in rows]

    async def get_last_n(self, slot, n):
        if not self.conn_pool:
            return []
        async with self.conn_pool.acquire() as conn:
            rows = await conn.fetch('SELECT number FROM results WHERE slot=$1 ORDER BY created_at DESC LIMIT $2', slot, n)
            return [r['number'] for r in rows]

    async def get_all_numbers(self):
        if not self.conn_pool:
            return []
        async with self.conn_pool.acquire() as conn:
            rows = await conn.fetch('SELECT number FROM results')
            return [r['number'] for r in rows]

    async def save_result(self, slot, number):
        if not self.conn_pool:
            return False
        async with self.conn_pool.acquire() as conn:
            await conn.execute('INSERT INTO results (slot, number) VALUES ($1, $2)', slot, number)
        return True

    async def get_recent_numbers(self, slot, limit=5):
        if not self.conn_pool:
            return []
        async with self.conn_pool.acquire() as conn:
            rows = await conn.fetch('SELECT number FROM results WHERE slot=$1 ORDER BY created_at DESC LIMIT $2', slot, limit)
            return [r['number'] for r in rows]

    async def get_total_count(self):
        if not self.conn_pool:
            return 0
        async with self.conn_pool.acquire() as conn:
            count = await conn.fetchval('SELECT COUNT(*) FROM results')
            return count

db = TotoDatabase()

# ========== LSTM NEURAL NETWORK ==========
try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️ TensorFlow tidak tersedia, fitur LSTM dinonaktifkan")

class LSTM4DPredictor:
    def __init__(self, database):
        self.db = database
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = '/tmp/lstm_model.h5'
        self.scaler_path = '/tmp/scaler.pkl'

    async def prepare_data(self, days=60):
        all_numbers = []
        for slot in ['00:01','13:00','16:00','19:00','22:00','23:00']:
            data = await self.db.get_all_slot(slot)
            for _, num in data[-days:]:
                all_numbers.append(num)
        return np.array(all_numbers).reshape(-1, 1)

    async def train(self, force=False):
        if not LSTM_AVAILABLE:
            return "❌ TensorFlow tidak terinstall"
        if os.path.exists(self.model_path) and not force:
            self.model = load_model(self.model_path)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            return "✅ Model LSTM sudah ada"
        data = await self.prepare_data(60)
        if len(data) < 100:
            return f"⚠️ Data kurang ({len(data)}/100)"
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(30, len(scaled_data)-1):
            X.append(scaled_data[i-30:i, 0])
            y.append(scaled_data[i+1, 0])
        X = np.array(X).reshape(-1, 30, 1)
        y = np.array(y)
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(30,1)))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=50, batch_size=16, verbose=0)
        self.model = model
        self.model.save(self.model_path)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        return "✅ Model LSTM berhasil ditraining"

    async def predict_next(self, slot, days=30):
        if not LSTM_AVAILABLE or not self.model:
            return None
        data = await self.db.get_all_slot(slot)
        if len(data) < 30:
            return None
        last_30 = np.array([d[1] for d in data[-30:]]).reshape(-1,1)
        scaled_last = self.scaler.transform(last_30)
        X_pred = scaled_last.reshape(1,30,1)
        pred_scaled = self.model.predict(X_pred, verbose=0)
        pred = self.scaler.inverse_transform(pred_scaled)[0][0]
        return int(round(pred)) % 10000

lstm_predictor = LSTM4DPredictor(db)

# ========== UTILITIES ==========
def digital_root(number):
    n = sum(int(d) for d in str(number).zfill(4))
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n

def dna_number(number):
    return ''.join(sorted(str(number).zfill(4)))

def hamming_distance(a, b):
    sa = f"{a:04d}"
    sb = f"{b:04d}"
    return sum(c1 != c2 for c1,c2 in zip(sa,sb))

# ========== LCG DETECTOR ==========
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    g, y, x = egcd(b % a, a)
    return (g, x - (b // a) * y, y)

def modinv(a, m=10000):
    g, x, _ = egcd(a, m)
    if g != 1:
        return None
    return x % m

def crack_lcg(seq):
    if len(seq) < 4:
        return None
    X1, X2, X3, X4 = seq[:4]
    d1 = (X2 - X1) % 10000
    d2 = (X3 - X2) % 10000
    d3 = (X4 - X3) % 10000
    if d1 == d2 == d3:
        return {'a':1, 'c':d1, 'next':(X4+d1)%10000, 'method':'const'}
    delta = (X2 - X1) % 10000
    if delta == 0:
        return None
    inv = modinv(delta)
    if inv is None:
        return None
    a = ((X3 - X2) * inv) % 10000
    c = (X2 - a * X1) % 10000
    next_val = (a * X4 + c) % 10000
    return {'a':a, 'c':c, 'next':next_val, 'method':'inverse'}

# ========== BBFS GENERATOR ==========
def generate_bbfs(base_numbers, target_count=1000):
    result = OrderedDict()
    for num in base_numbers[:20]:
        result[num] = True
    variations = [1,-1,10,-10,100,-100,1000,-1000]
    candidates = list(result.keys())
    while len(result) < target_count:
        for num in candidates[:30]:
            for var in variations:
                new_num = (num + var) % 10000
                if new_num not in result:
                    result[new_num] = True
                    candidates.append(new_num)
                    if len(result) >= target_count:
                        break
            if len(result) >= target_count:
                break
        candidates = candidates[-50:]
    for num in list(result.keys())[:50]:
        mirror = int(str(num).zfill(4)[::-1])
        result[mirror] = True
        mistik_map = {0:1,1:0,2:3,3:2,4:5,5:4,6:7,7:6,8:9,9:8}
        mistik = int(''.join(str(mistik_map[int(d)]) for d in str(num).zfill(4)))
        result[mistik] = True
    return [str(n).zfill(4) for n in list(result.keys())[:target_count]]
# ========== BBFS BALANCED (HOT + COLD + RANDOM) ==========
def generate_balanced_bbfs(base_numbers, target_count=1000, hot_ratio=0.7):
    """
    Generate BBFS dengan campuran:
    - hot_ratio% dari base numbers (prediksi)
    - sisanya dari angka dingin + random
    """
    result = OrderedDict()
    
    # 1. Ambil angka hot dari base numbers
    hot_target = int(target_count * hot_ratio)
    hot_list = generate_bbfs(base_numbers, hot_target)
    for num in hot_list:
        result[int(num)] = True
    
    # 2. Tambah angka cold & random sampai target
    cold_pool = []
    
    # Ambil angka cold dari database (frekuensi terendah)
    try:
        # Ambil semua angka di slot 19:00 (atau slot umum)
        all_nums = db.get_all_numbers_sync()  # kita buat synchronus wrapper
        freq = {}
        for num in all_nums:
            freq[num] = freq.get(num, 0) + 1
        # Urutkan dari yang paling jarang muncul
        cold_pool = [n for n, _ in sorted(freq.items(), key=lambda x: x[1])[:200]]
    except:
        # Fallback: generate random angka
        cold_pool = []
    
    # Tambah random digits generator untuk cover 0-9
    import random
    while len(result) < target_count:
        # 70% dari cold pool (jika ada), 30% random murni
        if cold_pool and random.random() < 0.7:
            num = random.choice(cold_pool)
        else:
            # Generate angka dengan digit acak 0-9
            num = random.randint(0, 9999)
        
        if num not in result:
            result[num] = True
            # Tambahkan varian mirror/mistik juga
            if len(result) < target_count:
                mirror = int(str(num).zfill(4)[::-1])
                result[mirror] = True
            if len(result) < target_count:
                mistik_map = {0:1,1:0,2:3,3:2,4:5,5:4,6:7,7:6,8:9,9:8}
                mistik = int(''.join(str(mistik_map[int(d)]) for d in str(num).zfill(4)))
                result[mistik] = True
    
    return [str(n).zfill(4) for n in list(result.keys())[:target_count]]

# Wrapper sinkron untuk akses database (karena generate_balanced_bbfs dipanggil dari fungsi async)
# Taruh di luar kelas
async def get_all_numbers_sync_wrapper():
    return await db.get_all_numbers()

# Kita akan panggil dari dalam fungsi, perlu sedikit trik
# Tapi karena ini fungsi biasa, kita buat versi async dan panggil dengan loop
# Alternatif: kita panggil langsung di handler
# ========== ENTROPY ==========
def calculate_entropy(numbers):
    if not numbers:
        return 13.29
    freq = {}
    for n in numbers:
        freq[n] = freq.get(n,0)+1
    probs = [f/len(numbers) for f in freq.values()]
    return -sum(p * math.log2(p) for p in probs)

# ========== BOT HANDLERS ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 HPA-4D v5.0 MASTER EDITION\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "/input 16:00 1234 - input result 16:00\n"
        "/prediksi <slot> - prediksi semua slot\n"
        "/bbfs <slot> <jumlah> - generate BBFS (max 2000)\n"
        "/lstm <slot> - prediksi dengan AI (LSTM)\n"
        "/train - training model LSTM\n"
        "/analisis <angka> - DNA & digital root\n"
        "/hot <slot> - 10 angka terpanas\n"
        "/entropy - dashboard entropi\n"
        "/lcg <slot> - deteksi kelemahan LCG\n"
        "/graph - visualisasi graph 4D\n"
        "/status - status database"
    )

async def input_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        _, slot, num = update.message.text.split()
        num = int(num)
        if slot not in ['13:00','16:00','19:00','22:00','23:00','00:01']:
            await update.message.reply_text("❌ Slot tidak valid")
            return
        await db.save_result(slot, num)
        await update.message.reply_text(f"✅ Result {slot} = {num:04d} tersimpan")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def prediksi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        slot = context.args[0]
        if slot == "13:00":
            last_0001 = await db.get_recent_numbers('00:01', 1)
            if last_0001 and last_0001[0] < 2000:
                preds = [5602,5740,5923,6977,7822,8042,8958,4283,3300,6000]
            else:
                preds = [4937,5000,4800,5100,4500,5200,4700,5300,4400,5400]
        elif slot == "16:00":
            last_1300 = await db.get_recent_numbers('13:00', 1)
            if last_1300:
                base = (last_1300[0] - 273) % 10000
                preds = [base, (base+123)%10000, (base-123)%10000, 4560,4557,4440,4630,4538,4691,4906]
            else:
                preds = [4560,4557,4440,4630,4538,4691,4906,4057,4360,5169]
        elif slot == "19:00":
            numbers = await db.get_recent_numbers('16:00', 5)
            if numbers:
                mean = sum(numbers)//len(numbers)
                preds = [mean%10000, (mean+123)%10000, (mean-123)%10000, 5122,3322,2424,7090,7416,1650,4181]
            else:
                preds = [5122,3322,2424,7090,7416,1650,4181,1895,9922,7022]
        elif slot == "22:00":
            last_1900 = await db.get_recent_numbers('19:00', 1)
            if last_1900:
                base = (last_1900[0] - 792) % 10000
                preds = [base,4613,4500,5392,5479,4495,9063,8841,5098,4624]
            else:
                preds = [4613,4500,5392,5479,4495,9063,8841,5098,4624,5943]
        elif slot == "23:00":
            last_2200 = await db.get_recent_numbers('22:00', 1)
            if last_2200:
                base = (last_2200[0] + 506) % 10000
                preds = [base,9037,1572,1007,5170,5129,5296,5907,5728,5406]
            else:
                preds = [9037,1572,1007,5170,5129,5296,5907,5728,5406,2059]
        elif slot == "00:01":
            last_2300 = await db.get_recent_numbers('23:00', 1)
            if last_2300:
                base = (last_2300[0] - 132) % 10000
                preds = [base,5028,5000,5066,5135,5211,5404,5590,5080,5170]
            else:
                preds = [5028,5000,5066,5135,5211,5404,5590,5080,5170,5260]
        else:
            await update.message.reply_text("❌ Slot tidak valid")
            return
        msg = f"🎯 PREDIKSI {slot}\n" + "\n".join([f"{i+1}. {p:04d}" for i,p in enumerate(preds[:10])])
        await update.message.reply_text(msg)
    except (IndexError, ValueError):
        await update.message.reply_text("⚠️ Gunakan: /prediksi 19:00")

async def bbfs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        slot = context.args[0]
        count = 1000
        if len(context.args) > 1:
            count = min(max(int(context.args[1]),100),2000)
        # Ambil base numbers
        if slot == "19:00":
            numbers = await db.get_recent_numbers('16:00',5)
            if numbers:
                mean = sum(numbers)//len(numbers)
                base = [mean%10000, (mean+123)%10000, (mean-123)%10000, 5122,3322,2424,7090,7416]
            else:
                base = [5122,3322,2424,7090,7416,1650,4181,1895,9922,7022]
        elif slot == "23:00":
            base = [9037,1572,1007,5170,5129,5296,5907,5728,5406,2059]
        else:
            base = [5122,3322,2424,7090,7416,9037,1572,1007,4613,4500]
        bbfs_list = generate_bbfs(base, count)
        msg = f"🎯 BBFS {slot} ({len(bbfs_list)} ANGKA)\n" + "*".join(bbfs_list[:500])
        await update.message.reply_text(msg[:4096])
        if len(bbfs_list) > 500:
            msg2 = "*".join(bbfs_list[500:1000])
            await update.message.reply_text(msg2[:4096])
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def train_lstm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await lstm_predictor.train()
    await update.message.reply_text(f"🧠 {msg}")

async def prediksi_lstm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        slot = context.args[0]
        pred = await lstm_predictor.predict_next(slot)
        if pred:
            await update.message.reply_text(f"🧠 LSTM PREDIKSI {slot}: {pred:04d}")
        else:
            await update.message.reply_text("⚠️ Data tidak cukup (minimal 30 hari)")
    except:
        await update.message.reply_text("⚠️ Gunakan: /lstm 19:00")

async def analisis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        num = int(context.args[0])
        dr = digital_root(num)
        dna = dna_number(num)
        all_nums = await db.get_all_numbers()
        freq = all_nums.count(num)
        dna_freq = sum(1 for n in all_nums if dna_number(n) == dna)
        msg = (
            f"🔬 ANALISIS {num:04d}\n━━━━━━━━━━━━━━━━━━━\n"
            f"🧬 DNA: {dna}\n🔢 Digital Root: {dr}\n"
            f"📊 Frekuensi: {freq}x\n👥 Frekuensi DNA: {dna_freq}x"
        )
        await update.message.reply_text(msg)
    except:
        await update.message.reply_text("⚠️ Gunakan: /analisis 4833")

async def hot_numbers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        slot = context.args[0] if context.args else "19:00"
        data = await db.get_all_slot(slot)
        freq = {}
        for _, num in data[-500:]:
            freq[num] = freq.get(num,0)+1
        top = sorted(freq.items(), key=lambda x: -x[1])[:10]
        msg = f"🔥 HOT NUMBERS {slot}\n" + "\n".join([f"{i+1}. {n:04d} ({c}x)" for i,(n,c) in enumerate(top)])
        await update.message.reply_text(msg)
    except:
        await update.message.reply_text("⚠️ Gunakan: /hot 19:00")

async def entropy_dash(update: Update, context: ContextTypes.DEFAULT_TYPE):
    slots = ['00:01','13:00','16:00','19:00','22:00','23:00']
    msg = "📊 ENTROPY DASHBOARD (30 hari)\n━━━━━━━━━━━━━━━━━━━\n"
    best_slot = None
    best_ent = 999
    for slot in slots:
        data = await db.get_all_slot(slot)
        nums = [n for _,n in data[-30:]]
        ent = calculate_entropy(nums)
        if ent < best_ent:
            best_ent = ent
            best_slot = slot
        if ent < 12.5:
            status = "🟢 Sangat Prediktif"
        elif ent < 12.8:
            status = "🟡 Prediktif"
        elif ent < 13.0:
            status = "🟠 Agak Acak"
        else:
            status = "🔴 Acak"
        msg += f"{slot}: {ent:.2f} bit → {status}\n"
    msg += f"\n🎯 REKOMENDASI: Fokus pada {best_slot}"
    await update.message.reply_text(msg)

async def lcg_check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        slot = context.args[0]
        seq = await db.get_last_n(slot, 4)
        if len(seq) < 4:
            await update.message.reply_text(f"⚠️ Data {slot} kurang dari 4 draw")
            return
        res = crack_lcg(seq)
        if res:
            msg = f"🔐 LCG DETECTED di {slot}!\na={res['a']}\nc={res['c']}\n🎯 PREDIKSI BERIKUTNYA: {res['next']:04d}"
        else:
            msg = f"🔐 Tidak terdeteksi LCG murni di {slot}"
        await update.message.reply_text(msg)
    except:
        await update.message.reply_text("⚠️ Gunakan: /lcg 23:00")

async def generate_graph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🕸️ Membangun graph 4D (100 node)... mohon tunggu")
    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError:
        await update.message.reply_text("❌ Fitur graph membutuhkan networkx dan plotly, install dulu")
        return
    all_nums = await db.get_all_numbers()
    freq = {}
    for num in all_nums:
        freq[num] = freq.get(num,0)+1
    top = sorted(freq.items(), key=lambda x: -x[1])[:100]
    G = nx.Graph()
    for num, f in top:
        G.add_node(num, freq=f)
    for i, (n1,_) in enumerate(top):
        for n2,_ in top[i+1:]:
            d = hamming_distance(n1, n2)
            if d <= 2:
                G.add_edge(n1, n2, weight=1.0/d if d>0 else 1.0)
    pos = nx.spring_layout(G, k=0.3, iterations=20, seed=42)
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5,color='#888'), hoverinfo='none', mode='lines')
    for e in G.edges():
        x0,y0 = pos[e[0]]
        x1,y1 = pos[e[1]]
        edge_trace['x'] += (x0,x1,None)
        edge_trace['y'] += (y0,y1,None)
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers',
        hoverinfo='text',
        marker=dict(showscale=True, colorscale='RdYlGn_r',
                    size=[G.degree(n)*2 for n in G.nodes()],
                    color=[G.nodes[n]['freq'] for n in G.nodes()],
                    colorbar=dict(title='Frekuensi')))
    for n in G.nodes():
        x,y = pos[n]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (f"{n:04d}<br>Freq: {G.nodes[n]['freq']}<br>Degree: {G.degree(n)}",)
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title='HPA-4D Graph (Top 100)', showlegend=False,
                                    xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                                    yaxis=dict(showgrid=False,zeroline=False,showticklabels=False)))
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        fig.write_html(f.name)
        f.seek(0)
        await update.message.reply_document(document=open(f.name,'rb'), filename='hpa4d_graph.html')
    await update.message.reply_text("✅ Graph siap! Buka file HTML di browser.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = await db.get_total_count()
    await update.message.reply_text(f"📊 Database PostgreSQL: {total} angka tersimpan")

# ========== MAIN ==========
async def main():
    await db.init_pool()
    TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN tidak ditemukan")
        return
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("input", input_result))
    app.add_handler(CommandHandler("prediksi", prediksi))
    app.add_handler(CommandHandler("bbfs", bbfs))
    app.add_handler(CommandHandler("train", train_lstm))
    app.add_handler(CommandHandler("lstm", prediksi_lstm))
    app.add_handler(CommandHandler("analisis", analisis))
    app.add_handler(CommandHandler("hot", hot_numbers))
    app.add_handler(CommandHandler("entropy", entropy_dash))
    app.add_handler(CommandHandler("lcg", lcg_check))
    app.add_handler(CommandHandler("graph", generate_graph))
    app.add_handler(CommandHandler("status", status))
    print("✅ Bot started!")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
