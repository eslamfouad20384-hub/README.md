import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import time
from datetime import datetime, timedelta

# ====== المجلدات ======
CACHE = "cache"
MODEL = "model"
TRADE_LOG = "trades.csv"
HISTORICAL = "historical_data"

for folder in [CACHE, MODEL, HISTORICAL]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ====== Rate Limiter ======
class RateLimiter:
    def __init__(self, calls_per_sec):
        self.calls_per_sec = calls_per_sec
        self.last = 0

    def wait(self):
        diff = time.time() - self.last
        if diff < 1/self.calls_per_sec:
            time.sleep((1/self.calls_per_sec) - diff)
        self.last = time.time()

# ====== خريطة أسماء العملات لكل مصدر ======
SYMBOL_MAP = {
    "USDT": {
        "Binance": "USDT", "CoinGecko": "tether", "CoinCap": "tether",
        "FreeCryptoAPI": "USDT", "DexScreener": "USDT"
    },
    "PAXG": {
        "Binance": "PAXG", "CoinGecko": "pax-gold", "CoinCap": "pax-gold",
        "FreeCryptoAPI": "PAXG", "DexScreener": "PAXG"
    },
    "XAUT": {
        "Binance": "XAUT", "CoinGecko": "tether-gold", "CoinCap": "tether-gold",
        "FreeCryptoAPI": "XAUT", "DexScreener": "XAUT"
    },
    "BTC": {
        "Binance": "BTC", "CoinGecko": "bitcoin", "CoinCap": "bitcoin",
        "FreeCryptoAPI": "BTC", "DexScreener": "BTC"
    },
    "ETH": {
        "Binance": "ETH", "CoinGecko": "ethereum", "CoinCap": "ethereum",
        "FreeCryptoAPI": "ETH", "DexScreener": "ETH"
    },
    "SOL": {
        "Binance": "SOL", "CoinGecko": "solana", "CoinCap": "solana",
        "FreeCryptoAPI": "SOL", "DexScreener": "SOL"
    },
    "ADA": {
        "Binance": "ADA", "CoinGecko": "cardano", "CoinCap": "cardano",
        "FreeCryptoAPI": "ADA", "DexScreener": "ADA"
    }
}

def map_symbol(symbol, source):
    if symbol in SYMBOL_MAP:
        return SYMBOL_MAP[symbol].get(source, symbol)
    return symbol

# ====== Source Wrappers ======
# Rate limiters لكل مصدر
bn_limiter = RateLimiter(4)
fc_limiter = RateLimiter(2)
cc_limiter = RateLimiter(4)
cg_limiter = RateLimiter(4)
ds_limiter = RateLimiter(2)

def fetch_binance(symbol, interval="1d", limit=100):
    bn_limiter.wait()
    symbol = map_symbol(symbol,"Binance")+"USDT"
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        r = requests.get(url, params=params, timeout=6).json()
        df = pd.DataFrame(r, columns=[
            "openTime","open","high","low","close","vol","closeTime",
            "qav","numTrades","tbBaseAv","tbQuoteAv","ignore"
        ])
        df["time"] = pd.to_datetime(df["openTime"], unit="ms")
        df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
        return df
    except: return None

def fetch_free_crypto(symbol):
    fc_limiter.wait()
    symbol = map_symbol(symbol,"FreeCryptoAPI")
    try:
        url = f"https://api.freecryptoapi.com/api/v1/market/history"
        params = {"symbol":symbol+"USDT","interval":"1d"}
        r = requests.get(url, params=params, timeout=6).json()
        data = r.get("data")
        if data:
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["openTime"], unit="ms")
            df[["open","high","low","close"]] = df[["open","high","low","close"]].astype(float)
            return df
    except: return None

def fetch_coin_cap(symbol):
    cc_limiter.wait()
    symbol = map_symbol(symbol,"CoinCap")
    try:
        url = f"https://api.coincap.io/v2/assets/{symbol}/history"
        params = {"interval":"d1"}
        r = requests.get(url, params=params, timeout=6).json()
        if "data" in r:
            df = pd.DataFrame(r["data"])
            df["time"] = pd.to_datetime(df["date"])
            df.rename(columns={"priceUsd":"close"}, inplace=True)
            df["open"]=df["close"]; df["high"]=df["close"]; df["low"]=df["close"]
            return df
    except: return None

def fetch_coin_gecko(symbol, days=30):
    cg_limiter.wait()
    symbol = map_symbol(symbol,"CoinGecko")
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {"vs_currency":"usd","days":days,"interval":"daily"}
        r = requests.get(url, params=params, timeout=6).json()
        if "prices" in r:
            df = pd.DataFrame(r["prices"], columns=["time","close"])
            df["time"]=pd.to_datetime(df["time"], unit="ms")
            df["open"]=df["close"]; df["high"]=df["close"]; df["low"]=df["close"]
            return df
    except: return None

def fetch_dex_screener(symbol):
    ds_limiter.wait()
    symbol = map_symbol(symbol,"DexScreener")
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{symbol}"
        r = requests.get(url, timeout=6).json()
        markets = r.get("pairs",[])
        if markets:
            df = pd.DataFrame(markets[0].get("history", []))
            if not df.empty:
                df["time"] = pd.to_datetime(df["timestamp"], unit="s")
                df.rename(columns={"close":"close"}, inplace=True)
                return df
    except: return None

SOURCES = [
    ("Binance", fetch_binance),
    ("FreeCryptoAPI", fetch_free_crypto),
    ("CoinCap", fetch_coin_cap),
    ("CoinGecko", fetch_coin_gecko),
    ("DexScreener", fetch_dex_screener)
]

def fetch_best_data(symbol, interval="1d"):
    for name, func in SOURCES:
        try:
            df = func(symbol)
            if df is not None and not df.empty:
                return df
        except:
            continue
    return pd.DataFrame()

# ====== المؤشرات ======
def add_indicators(df):
    df["close"]=df["close"].astype(float)
    df["high"]=df["high"].astype(float)
    df["low"]=df["low"].astype(float)
    df["EMA50"]=df["close"].ewm(span=50).mean()
    df["EMA200"]=df["close"].ewm(span=200).mean()
    df["prev_close"]=df["close"].shift(1)
    df["tr1"]=df["high"]-df["low"]
    df["tr2"]=abs(df["high"]-df["prev_close"])
    df["tr3"]=abs(df["low"]-df["prev_close"])
    df["TR"]=df[["tr1","tr2","tr3"]].max(axis=1)
    df["ATR"]=df["TR"].ewm(alpha=1/14, adjust=False).mean()
    df["return"]=df["close"].pct_change()
    return df.dropna()

# ====== AI Model ======
def train_ai(df, symbol):
    df["target"]=(df["close"].shift(-3)>df["close"]).astype(int)
    df=df.dropna()
    if len(df)<100: return 0
    X=df[["EMA50","EMA200","ATR","return"]]; y=df["target"]
    model_file=os.path.join(MODEL,f"{symbol}.pkl")
    if os.path.exists(model_file):
        model=pickle.load(open(model_file,"rb"))
    else:
        model=RandomForestClassifier(n_estimators=100,max_depth=5)
    try:
        model.fit(X,y)
        pickle.dump(model,open(model_file,"wb"))
        return model.predict_proba(X.iloc[-1:])[0][1]
    except:
        return 0

# ====== Signal + Logging ======
def log_trade(trade):
    if not os.path.exists(TRADE_LOG):
        df=pd.DataFrame(columns=list(trade.keys()))
        df=df.append(trade,ignore_index=True)
        df.to_csv(TRADE_LOG,index=False)
    else:
        df=pd.read_csv(TRADE_LOG)
        df=df.append(trade,ignore_index=True)
        df.to_csv(TRADE_LOG,index=False)

def market_condition(symbol):
    df=fetch_best_data(symbol,"1d")
    if df.empty: return "غير متاح"
    df=add_indicators(df)
    last=df.iloc[-1]
    if last["close"]>last["EMA50"]>last["EMA200"]: return "صاعد"
    elif last["close"]<last["EMA50"]<last["EMA200"]: return "هابط"
    else: return "عرضي"

def generate_signal(symbol):
    df4h=fetch_best_data(symbol,"4h")
    if df4h.empty: return {"العملة":symbol,"حالة الإشارة":"مرفوض","سبب":"بيانات 4H غير متاحة"}
    df4h=add_indicators(df4h)
    dfd=fetch_best_data(symbol,"1d")
    if dfd.empty: return {"العملة":symbol,"حالة الإشارة":"مرفوض","سبب":"بيانات يومية غير متاحة"}
    dfd=add_indicators(dfd)
    last=df4h.iloc[-1]
    if last["close"]<dfd["EMA50"].iloc[-1] and last["close"]<dfd["EMA200"].iloc[-1]:
        return {"العملة":symbol,"حالة الإشارة":"مرفوض","سبب":"السعر تحت EMA50 و EMA200 يومي"}
    prob=train_ai(df4h,symbol)
    entry=last["close"]; atr=last["ATR"]; stop=entry-atr*1.2; target=entry+atr*1.8
    trade_status="مقبول" if prob>=0.55 else "مرفوض"
    reason="" if trade_status=="مقبول" else f"قوة AI ضعيفة ({round(prob*100,2)}%)"
    trade={"العملة":symbol,"تاريخ":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "دخول":round(entry,4) if trade_status=="مقبول" else np.nan,
           "وقف":round(stop,4) if trade_status=="مقبول" else np.nan,
           "هدف":round(target,4) if trade_status=="مقبول" else np.nan,
           "احتمال_الصعود":round(prob*100,2),"حالة_السوق":market_condition(symbol),
           "حالة الإشارة":trade_status,"سبب":reason}
    log_trade(trade)
    return trade

def scan_market():
    coins = ["BTC","ETH","USDT","PAXG","XAUT","SOL","ADA"]
    results=[]
    for c in coins:
        results.append(generate_signal(c))
        time.sleep(0.3)
    df=pd.DataFrame(results)
    df.index=np.arange(1,len(df)+1)
    return df

# ====== Streamlit UI ======
st.markdown('<h4 style="font-size:16px;">AI Spot Scanner</h4>', unsafe_allow_html=True)
coins = ["BTC","ETH","USDT","PAXG","XAUT","SOL","ADA"]
st.markdown(f"### 🧭 حالة السوق العام: صاعد/هابط/عرضي حسب أغلبية العملات")

def highlight_rows(row):
    color='background-color: #d4f8d4' if row.get('حالة الإشارة')=='مقبول' else 'background-color: #f8d4d4'
    return [color]*len(row)

if st.button("🔍 فحص السوق مرة أخرى"):
    df=scan_market()
    st.dataframe(df.style.apply(highlight_rows, axis=1))
    if (df["حالة الإشارة"]=="مقبول").any():
        st.success("تم تسجيل الصفقات وتحسين ترتيب الإشارات!")
    else:
        st.info("لم يتم تسجيل أي صفقة جديدة، لكن تم فحص السوق!")
