"""Today's buy-limit levels for the tested rule: limit = last daily close - k * ATR14, only if close > SMA200.
Run fetch.py 1d first to refresh data."""
import json, urllib.request
import pandas as pd
import backtest as bt
from limit_robust import trades_for

for sym in bt.COINS:
    df = bt.load(sym, "1d")
    x = bt.indicators(df, 1)
    c, atr, sma = df.close.iloc[-1], x.atr.iloc[-1], x.sma200.iloc[-1]
    live = json.load(urllib.request.urlopen(f"https://data-api.binance.vision/api/v3/ticker/24hr?symbol={sym}"))
    print(f"{sym[:-4]:5} close {c:>10.4g}  ATR {atr / c * 100:4.1f}%  {'ABOVE' if c > sma else 'below'} 200d SMA ({sma:.4g})"
          f"  | limit 2ATR {c - 2 * atr:.4g}  2.5ATR {c - 2.5 * atr:.4g}  3ATR {c - 3 * atr:.4g}"
          f"  | today low so far {float(live['lowPrice']):.4g}")

print("\nfills per year, all 5 coins (daily, hold 3d):")
for k, f in ((2, "none"), (2, "own200"), (2.5, "own200"), (3, "own200")):
    t = pd.DataFrame(sum((trades_for(s, k, 3, f) for s in bt.COINS), []))
    print(f"  k={k} {f:6}: {len(t) / 8.2:4.1f}/yr   2026 trades: "
          + ", ".join(f"{r.coin} {r.date:%m-%d} {r.ret * 100:+.1f}%" for r in t[t.year == 2026].itertuples()))
