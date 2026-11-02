"""Daily dip-buy bot (Binance spot, long only), the rule tested in limit_robust.py:
if the last daily close is above its 200-day SMA, rest a buy-limit at close - K * ATR14 for the day;
if it fills, market-sell that amount at the daily open HOLD_DAYS later. No stop loss (stops hurt in the test).

Run once a day at 00:05 UTC (systemd timer in deploy/). No dependencies beyond Python 3.9+.
    python3 bot.py           paper trading on live Binance prices (default, state in paper_state.json)
    python3 bot.py --live    real orders; disabled until LIVE_TRADING = True; needs env BINANCE_API_KEY and BINANCE_API_SECRET
Each run's report goes to Telegram if env TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set (BOT_NAME = first line).
First live runs: set ORDER_USDT small (min order is 5 USDT/USDC). The spot testnet can't be used: it keeps
too little daily history for the 200-day SMA, so no orders would ever be placed there.
The API key needs spot trading enabled and withdrawals disabled.
The bot only ever sells what it bought, so coins you already hold are left alone.
"""
import bisect, hashlib, hmac, json, os, sys, time, urllib.error, urllib.parse, urllib.request
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN

COINS = ["BTCUSDT", "ZECUSDT", "XRPUSDT", "AVAXUSDT", "ADAUSDT", "NEARUSDC"]  # BTC weakest in the backtest; NEAR not backtested
K = 2.5             # limit = close - K * ATR14 (3.0 = fewer, bigger trades)
HOLD_DAYS = 3
ORDER_USDT = 100.0  # spent per buy order in the pair's quote coin (USDT, or USDC for NEARUSDC); keep enough free
LIVE_TRADING = False  # True lets --live place real orders; while False the bot only paper trades and alerts
FEE = 0.001         # paper mode only
STATE_DIR = os.environ.get("STATE_DIRECTORY", os.path.dirname(os.path.abspath(__file__)))  # set by systemd StateDirectory=


def levels(candles):
    """(close, limit price) from closed daily candles [(day, o, h, l, c)], or None = no order today."""
    if len(candles) < 200:
        return None
    c = [x[4] for x in candles]
    tr = [max(h - l, abs(h - pc), abs(l - pc)) for (_, _, h, l, _), pc in zip(candles[-14:], c[-15:-1])]
    close, sma200, atr = c[-1], sum(c[-200:]) / 200, sum(tr) / 14
    return (close, close - K * atr) if close > sma200 else None


def step(ex, state, path=None, log=print):
    """One daily run. Each coin is either waiting on yesterday's order, holding, or gets a new order."""
    today = ex.today.isoformat()
    for sym in COINS:
        s = state.setdefault(sym, {})
        try:
            if "order" in s:
                if s["order"]["day"] == today:
                    continue  # already ran today
                qty, px = ex.close_order(sym, s["order"])  # keeps what filled, cancels the rest
                if qty > 0:
                    s["pos"] = {"qty": qty, "px": px, "day": s["order"]["day"]}
                    sell_day = date.fromisoformat(s["order"]["day"]) + timedelta(days=HOLD_DAYS)
                    log(f"{sym} FILLED: bought {qty:g} @ {px:g} on {s['order']['day']}; sell at market on {sell_day} 00:05 UTC")
                else:
                    log(f"{sym} buy-limit @ {s['order']['price']:g} NOT filled: cancel it")
                del s["order"]
            if "pos" in s:
                p = s["pos"]
                if ex.today >= date.fromisoformat(p["day"]) + timedelta(days=HOLD_DAYS):
                    q, px = ex.sell_market(sym, p["qty"])
                    log(f"{sym} SELL {q:g} at market now: open {px:g} vs buy {p['px']:g} ({(px / p['px'] - 1) * 100:+.2f}% before fees)")
                    s.setdefault("closed", []).append({**p, "exit_px": px, "exit_day": today})
                    del s["pos"]
                continue  # as tested: no new order on a day that started in a position
            if s.get("closed") and s["closed"][-1]["exit_day"] == today:
                continue  # second run on a sell day: same rule as above
            lv = levels(ex.closed_daily(sym))
            if lv is None:
                log(f"{sym} below 200d SMA (or <200 days of data): no order")
                continue
            close, price = lv
            s["order"] = {**ex.buy_limit(sym, ORDER_USDT / price, price), "day": today, "price": price, "qty": ORDER_USDT / price}
            log(f"{sym} BUY-LIMIT {ORDER_USDT / price:g} @ {price:g} = {ORDER_USDT:g} {sym[-4:]} "
                f"({(1 - price / close) * 100:.1f}% below close {close:g}). Cancel if unfilled at {ex.today + timedelta(days=1)} 00:05 UTC; "
                f"if filled, sell at market on {ex.today + timedelta(days=HOLD_DAYS)} 00:05 UTC")
        except Exception as e:  # one coin failing must not stop the others; state stays as it was
            log(f"{sym} ERROR {e}")
        finally:
            if path:
                save(path, state)


def save(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written state file


def telegram(text):
    """Send text to env TELEGRAM_CHAT_ID; skipped when TELEGRAM_BOT_TOKEN is unset. Retries, then raises."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        return print("TELEGRAM_BOT_TOKEN not set: no alert sent", file=sys.stderr)
    data = urllib.parse.urlencode({"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text[:4096]}).encode()
    for attempt in range(3):
        try:
            return urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=30).close()
        except urllib.error.HTTPError as e:
            err = f"{e.code} {e.read().decode()}"
        except OSError as e:
            err = str(e)
        time.sleep(10)
    raise RuntimeError(f"telegram send failed: {err}")  # never the URL: it holds the token


def candle(r):
    return (datetime.fromtimestamp(r[0] / 1000, timezone.utc).date(), *map(float, r[1:5]))


class Paper:
    """Fills simulated from real daily candles, exactly as the backtest: a limit fills if that day's low
    reaches it, at min(open, limit); a sell fills at the day's open."""

    def __init__(self, candles, today):
        self.candles, self.today = candles, today  # {symbol: [(day, o, h, l, c), ...]} sorted by day
        self.days = {s: [x[0] for x in cs] for s, cs in candles.items()}

    @classmethod
    def from_api(cls, base="https://data-api.binance.vision"):
        get = lambda s: json.load(urllib.request.urlopen(f"{base}/api/v3/klines?symbol={s}&interval=1d&limit=300", timeout=30))
        return cls({s: [candle(r) for r in get(s)] for s in COINS}, datetime.now(timezone.utc).date())

    def _bar(self, sym, day):
        return self.candles[sym][bisect.bisect_left(self.days[sym], day)]

    def closed_daily(self, sym):
        i = bisect.bisect_left(self.days[sym], self.today)
        return self.candles[sym][max(0, i - 300):i]

    def buy_limit(self, sym, qty, price):
        return {}

    def close_order(self, sym, order):
        _, o, _, l, _ = self._bar(sym, date.fromisoformat(order["day"]))
        return (order["qty"] * (1 - FEE), min(o, order["price"])) if l <= order["price"] else (0.0, 0.0)

    def sell_market(self, sym, qty):
        return qty, self._bar(sym, self.today)[1]


def sign(secret, query):
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def floor(x, step):
    return (Decimal(str(x)) / step).to_integral_value(ROUND_DOWN) * step


def fmt(d):
    return format(d.normalize(), "f")


class Binance:
    def __init__(self, base, key=None, secret=None):
        self.base, self.key, self.secret, self.info = base, key, secret, {}
        self.today = datetime.now(timezone.utc).date()

    def _req(self, method, path, signed=False, **params):
        if signed:
            params.update(timestamp=int(time.time() * 1000), recvWindow=5000)
        q = urllib.parse.urlencode(params)
        if signed:
            q += "&signature=" + sign(self.secret, q)
        req = urllib.request.Request(f"{self.base}{path}?{q}", method=method,
                                     headers={"X-MBX-APIKEY": self.key} if self.key else {})
        try:
            return json.load(urllib.request.urlopen(req, timeout=30))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{method} {path} -> {e.code} {e.read().decode()}") from None

    def _info(self, sym):
        """(base asset, tick size, step size, min notional) from the exchange's symbol filters."""
        if sym not in self.info:
            s = self._req("GET", "/api/v3/exchangeInfo", symbol=sym)["symbols"][0]
            f = {x["filterType"]: x for x in s["filters"]}
            self.info[sym] = (s["baseAsset"], Decimal(f["PRICE_FILTER"]["tickSize"]), Decimal(f["LOT_SIZE"]["stepSize"]),
                              Decimal((f.get("NOTIONAL") or f["MIN_NOTIONAL"])["minNotional"]))
        return self.info[sym]

    def closed_daily(self, sym):
        rows = self._req("GET", "/api/v3/klines", symbol=sym, interval="1d", limit=300)
        return [x for x in map(candle, rows) if x[0] < self.today]

    def buy_limit(self, sym, qty, price):
        _, tick, step, min_notional = self._info(sym)
        p, q = floor(price, tick), floor(qty, step)
        if p * q < min_notional:
            raise RuntimeError(f"order {fmt(q)} @ {fmt(p)} is below min notional {min_notional}; raise ORDER_USDT")
        r = self._req("POST", "/api/v3/order", True, symbol=sym, side="BUY", type="LIMIT",
                      timeInForce="GTC", quantity=fmt(q), price=fmt(p))
        return {"id": r["orderId"]}

    def close_order(self, sym, order):
        try:  # cancel first, then read: nothing can fill between reading and cancelling
            self._req("DELETE", "/api/v3/order", True, symbol=sym, orderId=order["id"])
        except RuntimeError as e:
            if "-2011" not in str(e):  # -2011 = already filled or cancelled
                raise
        o = self._req("GET", "/api/v3/order", True, symbol=sym, orderId=order["id"])
        qty = float(o["executedQty"])
        if qty == 0:
            return 0.0, 0.0
        base = self._info(sym)[0]
        fills = self._req("GET", "/api/v3/myTrades", True, symbol=sym, orderId=order["id"])
        fee = sum(float(t["commission"]) for t in fills if t["commissionAsset"] == base)  # fee taken in the coin
        return qty - fee, float(o["cummulativeQuoteQty"]) / qty

    def sell_market(self, sym, qty):
        base, _, step, _ = self._info(sym)
        free = next(float(b["free"]) for b in self._req("GET", "/api/v3/account", True)["balances"] if b["asset"] == base)
        r = self._req("POST", "/api/v3/order", True, symbol=sym, side="SELL", type="MARKET",
                      quantity=fmt(floor(min(qty, free), step)))
        return float(r["executedQty"]), float(r["cummulativeQuoteQty"]) / float(r["executedQty"])


if __name__ == "__main__":
    live = "--live" in sys.argv
    if live and not LIVE_TRADING:
        sys.exit("live trading is disabled: set LIVE_TRADING = True in bot.py to place real orders")
    lines = [f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC  {'LIVE' if live else 'paper (no real orders)'}  K={K}  {ORDER_USDT:g}/order"]
    try:
        if live:
            ex = Binance(os.environ.get("BINANCE_BASE_URL", "https://api.binance.com"),
                         os.environ["BINANCE_API_KEY"], os.environ["BINANCE_API_SECRET"])
        else:
            ex = Paper.from_api()
        path = os.path.join(STATE_DIR, "live_state.json" if live else "paper_state.json")
        state = json.load(open(path)) if os.path.exists(path) else {}
        step(ex, state, path, log=lines.append)
    except Exception as e:  # e.g. Binance unreachable: still alert, then fail the systemd run
        lines.append(f"RUN FAILED: {e!r}")
        raise
    finally:
        print("\n".join(lines))
        telegram(os.environ.get("BOT_NAME", "dipbuy") + "\n\n" + "\n".join(lines))
