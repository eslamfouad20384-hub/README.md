# AI Spot Scanner

برنامج تحليل سوق العملات الرقمية باستخدام عدة مصادر بيانات مجانية، مع إشارات AI للتداول.

## المميزات
- دمج بيانات من Binance, FreeCryptoAPI, CoinCap, CoinGecko, DexScreener.
- دعم العملات: BTC, ETH, USDT, PAXG, XAUT, SOL, ADA.
- معالجة اختلاف أسماء العملات بين المصادر.
- إشارات AI مع تسجيل الصفقات في `trades.csv`.
- واجهة Streamlit تفاعلية ملونة حسب حالة الإشارة.
- بيانات فريم 4H و Daily.

## التشغيل
1. تثبيت المتطلبات:
```bash
pip install -r requirements.txt
