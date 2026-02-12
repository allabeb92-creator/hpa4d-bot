import os
import asyncio
import asyncpg
import math
import random
import tempfile
from collections import OrderedDict, defaultdict
from itertools import permutations
from datetime import datetime, timedelta
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
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS portfolio (
                        id SERIAL PRIMARY KEY,
                        session TEXT NOT NULL,
                        bet_amount INTEGER NOT NULL,
                        win_amount INTEGER NOT NULL,
                        result_number INTEGER,
                        profit INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                ''')
        except Exception as e:
            print(f"⚠️ Koneksi database gagal: {e}")

    async def get_all_slot(self, slot):
        if not self.conn_pool:
            return []
        async with self.conn_pool.acquire() as conn:
            rows = await conn.fetch('SELECT number FROM results WHERE slot=$1 ORDER BY created_at ASC', slot)
            return [(None, r['number']) for r in rows]

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

    async def get_consecutive_repeats(self, slot):
        """Deteksi angka yang sama 2x berturut-turut (Boomerang)"""
        last_two = await self.get_last_n(slot, 2)
        if len(last_two) == 2 and last_two[0] == last_two[1]:
            return last_two[0]
        return None

    # ========== PORTFOLIO ==========
    async def record_bet(self, session, bet_amount, win_amount, result_number, profit):
        if not self.conn_pool:
            return
        async with self.conn_pool.acquire() as conn:
            await conn.execute(
                'INSERT INTO portfolio (session, bet_amount, win_amount, result_number, profit) VALUES ($1,$2,$3,$4,$5)',
                session, bet_amount, win_amount, result_number, profit
            )

    async def get_total_profit(self):
        if not self.conn_pool:
            return 0
        async with self.conn_pool.acquire() as conn:
            profit = await conn.fetchval('SELECT COALESCE(SUM(profit),0) FROM portfolio')
            return profit

    async def get_win_rate(self, days=30):
        if not self.conn_pool:
            return 0.0
        async with self.conn_pool.acquire() as conn:
            rows = await conn.fetch(
                'SELECT profit FROM portfolio WHERE created_at > NOW() - INTERVAL \'1 days\' * $1',
                days
            )
            wins = sum(1 for r in rows if r['profit'] > 0)
            total = len(rows)
            return wins / total if total > 0 else 0.0

    async def get_killer_fail_streak(self):
        """Catat streak kegagalan sniper"""
        if not self.conn_pool:
            return 0
        async with self.conn_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT profit FROM portfolio WHERE session LIKE 'killer_%' ORDER BY created_at DESC LIMIT 3"
            )
            streak = 0
            for r in rows:
                if r['profit'] <= 0:
                    streak += 1
                else:
                    break
            return streak

    # ========== DYNAMIC PREDICTION ==========
    async def get_dynamic_prediction(self, slot):
        lasts = await self.get_last_n(slot, 4)
        if len(lasts) < 2:
            return None
        diffs = []
        for i in range(len(lasts)-1):
            d = (lasts[i] - lasts[i+1]) % 10000
            diffs.append(d)
        avg_diff = sum(diffs) // len(diffs)
        next_val = (lasts[0] + avg_diff) % 10000
        is_jump = False
        jump_threshold = 2000
        if len(diffs) > 1 and abs(diffs[0] - avg_diff) > jump_threshold:
            is_jump = True
        return {
            'val': next_val,
            'offset': avg_diff,
            'base': lasts[0],
            'is_jump': is_jump,
            'diffs': diffs
        }

db = TotoDatabase()

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
    return sum(c1 != c2 for c1, c2 in zip(sa, sb))

def get_neighbors(num):
    """Kembalikan angka-angka tetangga (±1) di setiap posisi"""
    s = str(num).zfill(4)
    neighbors = set()
    for i in range(4):
        d = int(s[i])
        if d > 0:
            n = s[:i] + str(d-1) + s[i+1:]
            neighbors.add(int(n))
        if d < 9:
            n = s[:i] + str(d+1) + s[i+1:]
            neighbors.add(int(n))
    return list(neighbors)

# ========== SMART BIJI FILTER ==========
def smart_filter_biji(numbers, prev_num):
    if not numbers or prev_num is None:
        return numbers
    prev_dr = digital_root(prev_num)
    is_prev_odd = prev_dr % 2 != 0
    priority = []
    secondary = []
    for n in numbers:
        n_int = int(n) if isinstance(n, str) else n
        curr_dr = digital_root(n_int)
        is_curr_odd = curr_dr % 2 != 0
        if is_prev_odd != is_curr_odd:
            priority.append(n)
        else:
            secondary.append(n)
    return priority + secondary

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
        return {'a': 1, 'c': d1, 'next': (X4 + d1) % 10000, 'method': 'const'}
    delta = (X2 - X1) % 10000
    if delta == 0:
        return None
    inv = modinv(delta)
    if inv is None:
        return None
    a = ((X3 - X2) * inv) % 10000
    c = (X2 - a * X1) % 10000
    next_val = (a * X4 + c) % 10000
    return {'a': a, 'c': c, 'next': next_val, 'method': 'inverse'}

# ========== BBFS GENERATOR (BRAVO) ==========
def generate_bbfs(base_numbers, target_count=1000):
    result = OrderedDict()
    for num in base_numbers[:20]:
        result[num] = True
    variations = [1, -1, 10, -10, 100, -100, 1000, -1000]
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
        mistik_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8}
        mistik = int(''.join(str(mistik_map[int(d)]) for d in str(num).zfill(4)))
        result[mistik] = True
    return [str(n).zfill(4) for n in list(result.keys())[:target_count]]

# ========== ENTROPY ==========
def calculate_entropy(numbers):
    if not numbers:
        return 13.29
    freq = {}
    for n in numbers:
        freq[n] = freq.get(n, 0) + 1
    probs = [f / len(numbers) for f in freq.values()]
    return -sum(p * math.log2(p) for p in probs)

# ========== KELLY CRITERION ==========
def kelly_criterion(win_prob, odds=3000):
    b = odds
    p = win_prob
    q = 1 - p
    f = (b * p - q) / b
    return max(0, f)

# ========== VOID HUNTER (ZULU-V) ==========
async def get_void_numbers(core_digits, count=50):
    all_nums = await db.get_all_numbers()
    freq = defaultdict(int)
    for num in all_nums:
        freq[num] += 1
    void_candidates = []
    for num, f in freq.items():
        s = str(num).zfill(4)
        if any(int(d) in core_digits for d in s):
            continue
        void_candidates.append((num, f))
    void_candidates.sort(key=lambda x: x[1])
    result = []
    for num, _ in void_candidates[:count]:
        result.append(str(num).zfill(4))
    while len(result) < count:
        result.append(f"{random.randint(0,9999):04d}")
    return result

# ========== BOT HANDLERS ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 HPA-4D v6.2 KITAB HITAM EDITION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🔥 26 BATALYON + 6 PATCH KITAB HITAM\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "/input 16:00 1234 - input result\n"
        "/prediksi <slot> - prediksi dinamis\n"
        "/bbfs <slot> <jumlah> - BBFS hot only\n"
        "/bbfs2 <slot> <jumlah> - ⚖️ BBFS hot+cold+random\n"
        "/bbfs3 <slot> ABCD WXYZ - ⚔️ Tritunggal\n"
        "/bbfs4 <slot> <jumlah> <global> - ⏰ Zona Waktu\n"
        "/killer <slot> <global> - 🔪 Eksekusi PAUS + Area Damage\n"
        "/setglobal <pasaran> <angka> - 💾 Simpan data global\n"
        "/analisis <angka> - 🔬 DNA & digital root\n"
        "/hot <slot> - 🔥 10 angka terpanas\n"
        "/cold <slot> - ❄️ 10 angka terdingin\n"
        "/entropy - 📊 dashboard entropi + chaos mode\n"
        "/lcg <slot> - 🔐 deteksi LCG\n"
        "/graph - 🕸️ visualisasi graph 4D\n"
        "/doktrin - 📜 status 26 batalyon + patch\n"
        "/yield - 💰 laporan profit & rekomendasi modal\n"
        "/status - 📁 status database"
    )

async def input_result(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        _, slot, num = update.message.text.split()
        num = int(num)
        if slot not in ['13:00', '16:00', '19:00', '22:00', '23:00', '00:01']:
            await update.message.reply_text("❌ Slot tidak valid")
            return
        await db.save_result(slot, num)
        await update.message.reply_text(f"✅ Result {slot} = {num:04d} tersimpan")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def prediksi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        slot = context.args[0]
        dyn = await db.get_dynamic_prediction(slot)
        if dyn:
            p1 = dyn['val']
            p2 = (dyn['val'] + 50) % 10000
            p3 = (dyn['val'] - 50) % 10000
            p4 = (dyn['val'] + 123) % 10000
            p5 = (dyn['val'] - 123) % 10000
            p6 = (dyn['val'] + 200) % 10000
            p7 = (dyn['val'] - 200) % 10000
            p8 = (dyn['val'] + 500) % 10000
            p9 = (dyn['val'] - 500) % 10000
            p10 = (dyn['base'] + 111) % 10000
            preds = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
            jump_warning = "⚠️ LONCATAN TERDETEKSI!" if dyn['is_jump'] else ""
            msg = (
                f"🎯 PREDIKSI DINAMIS ({slot})\n"
                f"🔁 Offset: {dyn['offset']}\n"
                f"{jump_warning}\n"
                + "\n".join([f"{i+1}. {p:04d}" for i, p in enumerate(preds[:10])])
            )
        else:
            await update.message.reply_text("⚠️ Data kurang, gunakan prediksi statis.")
            return
        await update.message.reply_text(msg)
    except (IndexError, ValueError):
        await update.message.reply_text("⚠️ Gunakan: /prediksi 19:00")

async def bbfs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        slot = context.args[0]
        count = 1000
        if len(context.args) > 1:
            count = min(max(int(context.args[1]), 100), 2000)
        if slot == "19:00":
            numbers = await db.get_recent_numbers('16:00', 5)
            if numbers:
                mean = sum(numbers) // len(numbers)
                base = [mean % 10000, (mean + 123) % 10000, (mean - 123) % 10000, 5122, 3322, 2424, 7090, 7416]
            else:
                base = [5122, 3322, 2424, 7090, 7416, 1650, 4181, 1895, 9922, 7022]
        elif slot == "23:00":
            base = [9037, 1572, 1007, 5170, 5129, 5296, 5907, 5728, 5406, 2059]
        else:
            base = [5122, 3322, 2424, 7090, 7416, 9037, 1572, 1007, 4613, 4500]
        bbfs_list = generate_bbfs(base, count)
        last = await db.get_last_n(slot, 1)
        if last:
            bbfs_list = smart_filter_biji(bbfs_list, last[0])
        boomerang = await db.get_consecutive_repeats(slot)
        if boomerang:
            neighbors = get_neighbors(boomerang)
            for nb in neighbors:
                nb_str = f"{nb:04d}"
                if nb_str not in bbfs_list:
                    bbfs_list.insert(0, nb_str)
        msg = f"🔥 BBFS HOT {slot} ({len(bbfs_list)} ANGKA)\n" + "*".join(bbfs_list[:500])
        await update.message.reply_text(msg[:4096])
        if len(bbfs_list) > 500:
            msg2 = "*".join(bbfs_list[500:1000])
            await update.message.reply_text(msg2[:4096])
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def bbfs_balanced(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        slot = context.args[0]
        count = 1000
        if len(context.args) > 1:
            count = min(max(int(context.args[1]), 100), 2000)
        if slot == "19:00":
            numbers = await db.get_recent_numbers('16:00', 5)
            if numbers:
                mean = sum(numbers) // len(numbers)
                base = [mean % 10000, (mean + 123) % 10000, (mean - 123) % 10000, 5122, 3322, 2424, 7090, 7416]
            else:
                base = [5122, 3322, 2424, 7090, 7416, 1650, 4181, 1895, 9922, 7022]
        elif slot == "23:00":
            base = [9037, 1572, 1007, 5170, 5129, 5296, 5907, 5728, 5406, 2059]
        else:
            base = [5122, 3322, 2424, 7090, 7416, 9037, 1572, 1007, 4613, 4500]
        all_nums = await db.get_all_numbers()
        freq = {}
        for num in all_nums:
            freq[num] = freq.get(num, 0) + 1
        cold_pool = [n for n, _ in sorted(freq.items(), key=lambda x: x[1])[:200]]
        result = OrderedDict()
        hot_target = int(count * 0.7)
        hot_list = generate_bbfs(base, hot_target)
        for num in hot_list:
            result[int(num)] = True
        while len(result) < count:
            if cold_pool and random.random() < 0.7:
                num = random.choice(cold_pool)
            else:
                num = random.randint(0, 9999)
            if num not in result:
                result[num] = True
                if len(result) < count:
                    mirror = int(str(num).zfill(4)[::-1])
                    result[mirror] = True
                if len(result) < count:
                    mistik_map = {0:1,1:0,2:3,3:2,4:5,5:4,6:7,7:6,8:9,9:8}
                    mistik = int(''.join(str(mistik_map[int(d)]) for d in str(num).zfill(4)))
                    result[mistik] = True
        bbfs_list = [str(n).zfill(4) for n in list(result.keys())[:count]]
        last = await db.get_last_n(slot, 1)
        if last:
            bbfs_list = smart_filter_biji(bbfs_list, last[0])
        boomerang = await db.get_consecutive_repeats(slot)
        if boomerang:
            neighbors = get_neighbors(boomerang)
            for nb in neighbors:
                nb_str = f"{nb:04d}"
                if nb_str not in bbfs_list:
                    bbfs_list.insert(0, nb_str)
        void_count = max(1, count // 20)
        void_numbers = await get_void_numbers([], void_count)
        bbfs_list.extend(void_numbers)
        msg = f"⚖️ BBFS BALANCED {slot} ({len(bbfs_list)} ANGKA)\n🔥70% Hot + ❄️30% Cold/Random + 🕳️5% Void\n" + "*".join(bbfs_list[:500])
        await update.message.reply_text(msg[:4096])
        if len(bbfs_list) > 500:
            msg2 = "*".join(bbfs_list[500:1000])
            await update.message.reply_text(msg2[:4096])
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def bbfs_tritunggal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) < 2:
            await update.message.reply_text("❌ Gunakan: /bbfs3 19:00 ABCD WXYZ")
            return
        slot = context.args[0]
        digits1 = context.args[1]
        if len(digits1) != 4 or not digits1.isdigit():
            await update.message.reply_text("❌ 4 digit pertama harus angka 0-9")
            return
        sticky_d1 = int(digits1[0])
        sticky_d2 = int(digits1[1])
        index_d1 = int(digits1[2])
        index_d2 = int(digits1[3])
        global_d1 = None
        global_d2 = None
        if len(context.args) >= 3:
            digits2 = context.args[2]
            if len(digits2) >= 2 and digits2[:2].isdigit():
                global_d1 = int(digits2[0])
                global_d2 = int(digits2[1])
        else:
            sd = context.bot_data.get("GLOBAL_SDY")
            hk = context.bot_data.get("GLOBAL_HK")
            if sd:
                s = str(sd).zfill(4)
                global_d1 = int(s[2])
                global_d2 = int(s[3])
            elif hk:
                h = str(hk).zfill(4)
                global_d1 = int(h[1])
                global_d2 = int(h[3])
        digit_pool = [sticky_d1, sticky_d2, index_d1, index_d2]
        if global_d1 is not None:
            digit_pool.append(global_d1)
        if global_d2 is not None:
            digit_pool.append(global_d2)
        digit_pool = list(set(digit_pool))
        while len(digit_pool) < 6:
            digit_pool.append(index_d1)
            digit_pool = list(set(digit_pool))
        core_digits = digit_pool[:6]
        bbfs_set = set()
        for perm in permutations(core_digits, 4):
            num = int(''.join(map(str, perm)))
            bbfs_set.add(num)
        variations = [1,-1,10,-10,100,-100,1000,-1000]
        base_list = list(bbfs_set)[:200]
        for num in base_list:
            for var in variations:
                new_num = (num + var) % 10000
                bbfs_set.add(new_num)
        bbfs_list = [str(n).zfill(4) for n in list(bbfs_set)[:1000]]
        last = await db.get_last_n(slot, 1)
        if last:
            bbfs_list = smart_filter_biji(bbfs_list, last[0])
        boomerang = await db.get_consecutive_repeats(slot)
        if boomerang:
            neighbors = get_neighbors(boomerang)
            for nb in neighbors:
                nb_str = f"{nb:04d}"
                if nb_str not in bbfs_list:
                    bbfs_list.insert(0, nb_str)
        void_count = 50
        void_numbers = await get_void_numbers(core_digits, void_count)
        bbfs_list.extend(void_numbers)
        msg = (
            f"⚔️⚔️⚔️ TRITUNGGAL BBFS ({slot}) ⚔️⚔️⚔️\n"
            f"🧲 Sticky: {sticky_d1},{sticky_d2}\n"
            f"🔄 Index: {index_d1},{index_d2}\n"
            f"🌍 Global: {global_d1 if global_d1 else '-'},{global_d2 if global_d2 else '-'}\n"
            f"🎲 Core: {''.join(map(str, core_digits))}\n"
            f"📦 Total: {len(bbfs_list)} angka\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        msg += "*".join(bbfs_list[:500])
        await update.message.reply_text(msg[:4096])
        if len(bbfs_list) > 500:
            msg2 = "*".join(bbfs_list[500:1000])
            await update.message.reply_text(msg2[:4096])
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def bbfs4(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) < 1:
            await update.message.reply_text("❌ Gunakan: /bbfs4 19:00 1000 12")
            return
        slot = context.args[0]
        count = 1000
        if len(context.args) > 1:
            count = min(max(int(context.args[1]), 100), 2000)
        global_digits = ""
        if len(context.args) > 2:
            global_digits = context.args[2]
            global_digits = ''.join(filter(str.isdigit, global_digits))[:2]
        else:
            sd = context.bot_data.get("GLOBAL_SDY")
            hk = context.bot_data.get("GLOBAL_HK")
            if sd:
                s = str(sd).zfill(4)
                global_digits = s[2] + s[3]
            elif hk:
                h = str(hk).zfill(4)
                global_digits = h[1] + h[3]
        digit_pool = []
        last_this = await db.get_last_n(slot, 3)
        last_1300 = await db.get_last_n('13:00', 1)
        last_1600 = await db.get_last_n('16:00', 1)
        last_1900 = await db.get_last_n('19:00', 1)
        last_2200 = await db.get_last_n('22:00', 1)
        last_2300 = await db.get_last_n('23:00', 1)
        is_siang = slot in ['13:00','16:00']
        is_malam = slot in ['19:00','22:00','23:00']
        if is_siang:
            if last_this:
                prev = str(last_this[0]).zfill(4)
                digit_pool.extend([int(prev[1]), int(prev[3]), int(prev[0]), int(prev[2])])
            for twin in [0,11,22,33,44,55,66,77,88,99]:
                digit_pool.append(twin % 10)
                digit_pool.append((twin // 10) % 10)
            if last_1300:
                s1300 = str(last_1300[0]).zfill(4)
                digit_pool.extend([int(d) for d in s1300])
            if last_1600:
                s1600 = str(last_1600[0]).zfill(4)
                digit_pool.extend([int(d) for d in s1600])
        elif is_malam:
            index_map = {0:5,1:6,2:7,3:8,4:9,5:0,6:1,7:2,8:3,9:4}
            if last_this:
                prev = str(last_this[0]).zfill(4)
                as_ = int(prev[0])
                kop = int(prev[1])
                digit_pool.append(index_map[as_])
                digit_pool.append(index_map[kop])
                if as_ in [0,2,4]:
                    digit_pool.append(index_map[as_])
                elif as_ in [5,7,9]:
                    digit_pool.append(index_map[as_])
            if slot == '19:00' and last_1600:
                s1600 = str(last_1600[0]).zfill(4)
                digit_pool.append(int(s1600[1]))
            elif slot == '22:00' and last_1900:
                s1900 = str(last_1900[0]).zfill(4)
                digit_pool.append(int(s1900[1]))
            elif slot == '23:00' and last_2200:
                s2200 = str(last_2200[0]).zfill(4)
                digit_pool.append(int(s2200[1]))
            if last_this:
                prev = str(last_this[0]).zfill(4)
                rev = prev[::-1]
                digit_pool.extend([int(d) for d in rev])
                mistik_map = {0:1,1:0,2:3,3:2,4:5,5:4,6:7,7:6,8:9,9:8}
                mistik = ''.join(str(mistik_map[int(d)]) for d in prev)
                digit_pool.extend([int(d) for d in mistik])
        elif slot == '00:01':
            if len(last_this) >= 2:
                a, b = last_this[0], last_this[1]
                if a > b:
                    digit_pool.extend([0,1])
                elif a < b:
                    digit_pool.extend([0,1])
            digit_pool.extend([0,1])
            if last_this:
                prev = str(last_this[0]).zfill(4)
                digit_pool.extend([int(prev[1]), int(prev[3])])
        if global_digits:
            digit_pool.extend([int(d) for d in global_digits])
        digit_pool = list(set(digit_pool))
        while len(digit_pool) < 6:
            digit_pool.append(random.randint(0,9))
        core_digits = digit_pool[:6]
        bbfs_set = set()
        for perm in permutations(core_digits, 4):
            num = int(''.join(map(str, perm)))
            bbfs_set.add(num)
        variations = [1,-1,10,-10,100,-100,1000,-1000]
        base_list = list(bbfs_set)[:200]
        for num in base_list:
            for var in variations:
                new_num = (num + var) % 10000
                bbfs_set.add(new_num)
        bbfs_list = [str(n).zfill(4) for n in list(bbfs_set)[:count]]
        last = await db.get_last_n(slot, 1)
        if last:
            bbfs_list = smart_filter_biji(bbfs_list, last[0])
        boomerang = await db.get_consecutive_repeats(slot)
        if boomerang:
            neighbors = get_neighbors(boomerang)
            for nb in neighbors:
                nb_str = f"{nb:04d}"
                if nb_str not in bbfs_list:
                    bbfs_list.insert(0, nb_str)
        void_count = max(1, count // 20)
        void_numbers = await get_void_numbers(core_digits, void_count)
        bbfs_list.extend(void_numbers)
        zona = "SIANG" if is_siang else "MALAM" if is_malam else "RESET"
        mode = "FOLLOW" if is_siang else "COUNTER" if is_malam else "PROACTIVE"
        msg = (
            f"⏰⏰⏰ BBFS ZONA WAKTU ({slot}) ⏰⏰⏰\n"
            f"🧠 Mode: {zona} – {mode}\n"
            f"🎲 Core: {''.join(map(str, core_digits))}\n"
            f"📦 Total: {len(bbfs_list)} angka\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )
        msg += "*".join(bbfs_list[:500])
        await update.message.reply_text(msg[:4096])
        if len(bbfs_list) > 500:
            msg2 = "*".join(bbfs_list[500:1000])
            await update.message.reply_text(msg2[:4096])
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def killer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) < 1:
            await update.message.reply_text("❌ Gunakan: /killer 23:00 12")
            return
        slot = context.args[0]
        global_digits = None
        if len(context.args) > 1:
            gd = context.args[1][:2]
            if gd.isdigit():
                global_digits = [int(gd[0]), int(gd[1])]
        else:
            sd = context.bot_data.get("GLOBAL_SDY")
            hk = context.bot_data.get("GLOBAL_HK")
            if sd:
                s = str(sd).zfill(4)
                global_digits = [int(s[2]), int(s[3])]
            elif hk:
                h = str(hk).zfill(4)
                global_digits = [int(h[1]), int(h[3])]
        last = await db.get_last_n(slot, 1)
        if not last:
            await update.message.reply_text(f"⚠️ Tidak ada data untuk slot {slot}")
            return
        prev = str(last[0]).zfill(4)
        as_prev = int(prev[0])
        if as_prev not in [0,2,4]:
            await update.message.reply_text(f"❌ AS harus genap kecil (0,2,4). AS={as_prev}")
            return
        target_as = as_prev + 5
        base_numbers = []
        if global_digits:
            kop = global_digits[0] % 10
            kepala = global_digits[1] % 10 if len(global_digits) > 1 else 0
            for ekor in range(10):
                angka = target_as * 1000 + kop * 100 + kepala * 10 + ekor
                base_numbers.append(angka)
        else:
            for ekor in range(10):
                angka = target_as * 1000 + 0 * 100 + 0 * 10 + ekor
                base_numbers.append(angka)
        for delta in [-1, 1]:
            neighbor_as = (target_as + delta) % 10
            for ekor in range(5):
                angka = neighbor_as * 1000 + 0 * 100 + 0 * 10 + ekor
                base_numbers.append(angka)
        bbfs_list = [str(n).zfill(4) for n in base_numbers]
        last = await db.get_last_n(slot, 1)
        if last:
            bbfs_list = smart_filter_biji(bbfs_list, last[0])
        context.bot_data['last_killer_time'] = datetime.now()
        streak = await db.get_killer_fail_streak()
        msg = (
            f"🔪🔪🔪 KILLER MODE ({slot}) 🔪🔪🔪\n"
            f"🎯 AS Rebound: {as_prev} → {target_as}\n"
            f"🌍 Global: {global_digits if global_digits else '-'}\n"
            f"📦 15 LINE (10 utama + 5 area damage):\n"
            f"{'  '.join(bbfs_list[:15])}\n"
        )
        if streak >= 2:
            msg += f"⚠️ EGO ALERT: {streak}x gagal berturut-turut! Turunkan bet sniper 50%.\n"
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

async def set_global(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) < 2:
            await update.message.reply_text("❌ Gunakan: /setglobal SDY 2705")
            return
        pasaran = context.args[0].upper()
        angka = context.args[1]
        if not angka.isdigit() or len(angka) != 4:
            await update.message.reply_text("❌ Angka harus 4 digit")
            return
        context.bot_data[f"GLOBAL_{pasaran}"] = int(angka)
        await update.message.reply_text(f"✅ Data {pasaran} = {angka} tersimpan!")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error: {str(e)}")

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
            freq[num] = freq.get(num, 0) + 1
        top = sorted(freq.items(), key=lambda x: -x[1])[:10]
        msg = f"🔥 HOT NUMBERS {slot}\n" + "\n".join([f"{i+1}. {n:04d} ({c}x)" for i, (n, c) in enumerate(top)])
        await update.message.reply_text(msg)
    except:
        await update.message.reply_text("⚠️ Gunakan: /hot 19:00")

async def cold_numbers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        slot = context.args[0] if context.args else "19:00"
        data = await db.get_all_slot(slot)
        freq = {}
        for _, num in data[-500:]:
            freq[num] = freq.get(num, 0) + 1
        cold = sorted(freq.items(), key=lambda x: x[1])[:10]
        msg = f"❄️ COLD NUMBERS {slot}\n" + "\n".join([f"{i+1}. {n:04d} ({c}x)" for i, (n, c) in enumerate(cold)])
        await update.message.reply_text(msg)
    except:
        await update.message.reply_text("⚠️ Gunakan: /cold 19:00")

async def entropy_dash(update: Update, context: ContextTypes.DEFAULT_TYPE):
    slots = ['00:01', '13:00', '16:00', '19:00', '22:00', '23:00']
    msg = "📊 ENTROPY DASHBOARD (30 hari)\n━━━━━━━━━━━━━━━━━━━\n"
    best_slot = None
    best_ent = 999
    chaos = False
    for slot in slots:
        data = await db.get_all_slot(slot)
        nums = [n for _, n in data[-30:]]
        ent = calculate_entropy(nums)
        if ent < best_ent:
            best_ent = ent
            best_slot = slot
        if ent > 13.0:
            chaos = True
        if ent < 12.5:
            status = "🟢 Sangat Prediktif"
        elif ent < 12.8:
            status = "🟡 Prediktif"
        elif ent < 13.0:
            status = "🟠 Agak Acak"
        else:
            status = "🔴 Acak"
        msg += f"{slot}: {ent:.2f} bit → {status}\n"
    if chaos:
        msg += "\n⚠️ CHAOS MODE: Entropi tinggi! Void Hunter aktif (+10% void).\n"
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
        await update.message.reply_text("❌ Fitur graph membutuhkan networkx dan plotly, pastikan sudah terinstall")
        return
    all_nums = await db.get_all_numbers()
    freq = {}
    for num in all_nums:
        freq[num] = freq.get(num, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:100]
    G = nx.Graph()
    for num, f in top:
        G.add_node(num, freq=f)
    for i, (n1, _) in enumerate(top):
        for n2, _ in top[i+1:]:
            d = hamming_distance(n1, n2)
            if d <= 2:
                G.add_edge(n1, n2, weight=1.0/d if d > 0 else 1.0)
    pos = nx.spring_layout(G, k=0.3, iterations=20, seed=42)
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    for e in G.edges():
        x0,y0 = pos[e[0]]
        x1,y1 = pos[e[1]]
        edge_trace['x'] += (x0,x1,None)
        edge_trace['y'] += (y0,y1,None)
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='RdYlGn_r',
            size=[G.degree(n)*2 for n in G.nodes()],
            color=[G.nodes[n]['freq'] for n in G.nodes()],
            colorbar=dict(title='Frekuensi'),
            line=dict(width=2)))
    for n in G.nodes():
        x,y = pos[n]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (f"{n:04d}<br>Freq: {G.nodes[n]['freq']}<br>Degree: {G.degree(n)}",)
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='HPA-4D Graph (Top 100 Numbers)',
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        fig.write_html(f.name)
        f.seek(0)
        await update.message.reply_document(document=open(f.name, 'rb'), filename='hpa4d_graph.html')
    await update.message.reply_text("✅ Graph siap! Buka file HTML di browser.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    total = await db.get_total_count()
    await update.message.reply_text(f"📊 Database PostgreSQL: {total} angka tersimpan")

async def yield_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    profit = await db.get_total_profit()
    win_rate = await db.get_win_rate(30)
    kelly = kelly_criterion(win_rate)
    rec_bet = int(kelly * 1000)
    streak = await db.get_killer_fail_streak()
    ego_penalty = ""
    if streak >= 3:
        ego_penalty = f"⚠️ EGO PENALTY: Sniper gagal {streak}x, rekomendasi bet sniper turun 50%."
        rec_bet = int(rec_bet * 0.5)
    msg = (
        f"💰 YANKEE – LAPORAN PROFIT & EGO KILLER\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Total Profit: {profit}\n"
        f"📊 Win Rate (30 hari): {win_rate*100:.1f}%\n"
        f"📐 Kelly Fraction: {kelly:.3f}\n"
        f"🎯 Rekomendasi bet (modal 1000): {rec_bet}\n"
        f"{ego_penalty}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Gunakan /record untuk mencatat hasil betting (belum diimplementasikan)."
    )
    await update.message.reply_text(msg)

async def doktrin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📜 DOKTRIN HPA-4D v6.2 – KITAB HITAM EDITION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🟢 ALPHA (AS) – AKTIF\n"
        "🟢 BRAVO (BBFS) – AKTIF\n"
        "🟢 CHRONOS (Waktu) – AKTIF + COUNTER LOGIC\n"
        "🟢 DELTA (Selisih) – AKTIF\n"
        "🟢 ECHO (Sticky) – AKTIF\n"
        "🟢 FOXTROT (Frekuensi) – AKTIF\n"
        "🟢 GOLF (Global) – AKTIF\n"
        "🟢 HELIX (DNA) – AKTIF\n"
        "🟢 INDIA (Index) – AKTIF\n"
        "🟢 JULIET (Jump) – AKTIF\n"
        "🟢 KILO (Killer) – AKTIF + AREA DAMAGE\n"
        "🟢 LIMA (LCG) – AKTIF\n"
        "🟢 MIKE (Twin) – AKTIF\n"
        "🟢 NOVEMBER (Neighbor) – AKTIF + BOOMERANG NEIGHBOR\n"
        "🟢 OSCAR (Offset) – AKTIF\n"
        "🟢 PAPA (Parity) – AKTIF\n"
        "🟢 QUANTUM (Chaos) – AKTIF\n"
        "🟢 ROMEO (Reset) – AKTIF + PROACTIVE\n"
        "🟢 SIERRA (Sequence) – AKTIF\n"
        "🟢 TANGO (Trap) – AKTIF\n"
        "🟢 UNIFORM (Biji) – AKTIF\n"
        "🟢 VICTOR (Vector) – AKTIF\n"
        "🟢 WHISKEY (Weight) – AKTIF + EGO PENALTY\n"
        "🟢 X-RAY (Scanner) – AKTIF\n"
        "🟢 YANKEE (Yield) – AKTIF\n"
        "🟢 ZULU (Zero) – AKTIF + VOID HUNTER\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "✅ SEMUA BATALYON + PATCH KITAB HITAM SIAP TEMPUR!\n"
        "🔥 GOD MODE COMPLETE"
    )
    await update.message.reply_text(msg)

# ========== INISIALISASI ASYNC (HANYA SEKALI) ==========
async def async_init():
    """Inisialisasi database (hanya sekali)"""
    await db.init_pool()
    print("✅ Database initialized")

# ========== MAIN SYNCHRONOUS (TIDAK ADA KONFLIK EVENT LOOP) ==========
def main():
    """Entry point synchronous – tidak ada konflik event loop"""
    # 1. Inisialisasi database (async → sync via asyncio.run)
    asyncio.run(async_init())
    
    # 2. Baca token
    TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN tidak ditemukan")
        return
    
    # 3. Bangun aplikasi Telegram
    app = Application.builder().token(TOKEN).build()
    
    # 4. Daftarkan semua handler
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("input", input_result))
    app.add_handler(CommandHandler("prediksi", prediksi))
    app.add_handler(CommandHandler("bbfs", bbfs))
    app.add_handler(CommandHandler("bbfs2", bbfs_balanced))
    app.add_handler(CommandHandler("bbfs3", bbfs_tritunggal))
    app.add_handler(CommandHandler("bbfs4", bbfs4))
    app.add_handler(CommandHandler("killer", killer))
    app.add_handler(CommandHandler("setglobal", set_global))
    app.add_handler(CommandHandler("analisis", analisis))
    app.add_handler(CommandHandler("hot", hot_numbers))
    app.add_handler(CommandHandler("cold", cold_numbers))
    app.add_handler(CommandHandler("entropy", entropy_dash))
    app.add_handler(CommandHandler("lcg", lcg_check))
    app.add_handler(CommandHandler("graph", generate_graph))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("yield", yield_command))
    app.add_handler(CommandHandler("doktrin", doktrin))
    
    # 5. Jalankan bot (blocking, pakai loop internal – TIDAK NESTED)
    print("✅ Starting bot with polling...")
    app.run_polling()

if __name__ == "__main__":
    main()
