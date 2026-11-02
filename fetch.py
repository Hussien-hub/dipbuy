"""Download full Binance spot kline history (public endpoint, no key) to data/<SYMBOL>_<interval>.csv."""
import csv, json, os, sys, time, urllib.request

BASE = "https://data-api.binance.vision/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ZECUSDT", "XRPUSDT", "AVAXUSDT", "ADAUSDT", "NEARUSDC"]
INTERVALS = sys.argv[1:] or ["1d", "4h", "1h"]
OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

for sym in SYMBOLS:
    for iv in INTERVALS:
        rows, start = [], 0
        while True:
            url = f"{BASE}?symbol={sym}&interval={iv}&limit=1000&startTime={start}"
            batch = json.load(urllib.request.urlopen(url, timeout=30))
            if not batch:
                break
            rows += batch
            start = batch[-1][0] + 1
            if len(batch) < 1000:
                break
            time.sleep(0.05)
        rows = rows[:-1]  # drop the still-open candle
        with open(f"{OUT}/{sym}_{iv}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "open", "high", "low", "close", "volume"])
            w.writerows(r[:6] for r in rows)
        print(sym, iv, len(rows), "candles")
