"""The bot, replayed day by day on full history in paper mode, must make exactly the backtest's trades."""
from datetime import date, timedelta
from decimal import Decimal
import backtest as bt
import bot

assert bot.sign("NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j",  # Binance API docs example
                "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559"
                ) == "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"
assert bot.fmt(bot.floor(0.123456789, Decimal("0.00010000"))) == "0.1234"
assert bot.fmt(bot.floor(123.9, Decimal("1.00000000"))) == "123"

candles, frames = {}, {}
for sym in bot.COINS:
    df = bt.load(sym, "1d")
    frames[sym] = df
    candles[sym] = [(t.date(), r.open, r.high, r.low, r.close) for t, r in zip(df.index, df.itertuples())]

ex = bot.Paper(candles, None)
state = {}
day, last = min(c[0][0] for c in candles.values()), max(c[-1][0] for c in candles.values())
while day <= last:
    ex.today = day
    bot.step(ex, state, log=lambda *_: None)
    day += timedelta(days=1)

total = 0
for sym, df in frames.items():
    x = bt.indicators(df, 1)
    lim = (df.close - bot.K * x.atr).where(df.close > x.sma200)
    trades, _ = bt.run(df, lim, df.close < 0, bot.HOLD_DAYS, None, None)
    want = [(df.index[e].date().isoformat(), df.index[xx].date().isoformat(),
             min(df.open.iloc[e], lim.iloc[e - 1]), df.open.iloc[xx]) for e, xx, _ in trades]
    got = [(t["day"], t["exit_day"], t["px"], t["exit_px"]) for t in state[sym].get("closed", [])]
    assert len(got) == len(want), (sym, len(got), len(want))
    for g, w in zip(got, want):
        assert g[:2] == w[:2] and abs(g[2] / w[2] - 1) < 1e-9 and abs(g[3] / w[3] - 1) < 1e-9, (sym, g, w)
    total += len(got)
print(f"bot matches backtest: {total} trades across {len(frames)} coins")

# a second run on a sell day must not place an order (systemd restart, manual run)
for t in state["BTCUSDT"]["closed"]:
    ex.today = date.fromisoformat(t["exit_day"])
    fresh, again = {}, {"BTCUSDT": {"closed": [t]}}
    bot.step(ex, fresh, log=lambda *_: None)
    if "order" in fresh["BTCUSDT"]:
        bot.step(ex, again, log=lambda *_: None)
        assert "order" not in again["BTCUSDT"], again
        break
else:
    raise AssertionError("no sell day with a buy signal to test")
print("no re-entry on a sell day ok")
