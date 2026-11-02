"""Long-only spot dip-buy backtests on Binance data.

Signal on bar close -> fill at next bar open. Fee 0.1%/side + 0.05% slippage/side.
One position per coin, all-in/all-out of that coin's sleeve, cash earns 0.
In-sample (IS) = trades entered before SPLIT, out-of-sample (OOS) = after.
Edge = trade return minus the average return of a random entry held the same number of bars
in the same coin and period, so "buying any day" is the bar to beat.
"""
import os, sys
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "data")
COINS = ["BTCUSDT", "ZECUSDT", "XRPUSDT", "AVAXUSDT", "ADAUSDT"]
SPLIT = pd.Timestamp("2023-01-01")
FEE, SLIP = 0.001, 0.0005


def load(sym, iv):
    df = pd.read_csv(f"{DATA}/{sym}_{iv}.csv")
    df.index = pd.to_datetime(df.pop("ts"), unit="ms")
    return df


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def indicators(df, bars_per_day):
    c, h, l = df.close, df.high, df.low
    x = pd.DataFrame(index=df.index)
    x["sma5"], x["sma10"], x["sma20"] = c.rolling(5).mean(), c.rolling(10).mean(), c.rolling(20).mean()
    x["sma200"] = c.rolling(200 * bars_per_day).mean()  # 200-day trend regardless of timeframe
    x["rsi2"], x["rsi14"] = rsi(c, 2), rsi(c, 14)
    sd20 = c.rolling(20).std()
    x["bbl"] = x.sma20 - 2 * sd20
    x["z20"] = (c - x.sma20) / sd20
    r1 = c.pct_change()
    vol = r1.rolling(60 * bars_per_day).std()  # 60-day volatility, used to scale drops per coin
    x["dropz1"] = r1 / vol
    x["dropz3"] = c.pct_change(3 * bars_per_day) / (vol * np.sqrt(3 * bars_per_day))
    x["dropz7"] = c.pct_change(7 * bars_per_day) / (vol * np.sqrt(7 * bars_per_day))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    x["hi20"] = h.rolling(20).max()
    x["volr"] = df.volume / df.volume.rolling(20).mean()
    x["ibs"] = (c - l) / (h - l)
    x["mayer"] = c / x.sma200
    return x


def strategies(df, x, d):
    """d = bars per day; holds written as N * d are in days, bare numbers are bars."""
    c, o = df.close, df.open
    S = {}  # name -> (entry, exit, max_bars, tp, sl)
    S["RSI2<10 -> close>SMA5"] = (x.rsi2 < 10, c > x.sma5, 10, None, None)
    S["RSI2<5 -> close>SMA5"] = (x.rsi2 < 5, c > x.sma5, 10, None, None)
    S["RSI2<10 + green bar -> close>SMA5"] = ((x.rsi2.shift() < 10) & (c > o) & (c > c.shift()), c > x.sma5, 10, None, None)
    S["close<BB lower -> close>SMA20"] = (c < x.bbl, c > x.sma20, 20, None, None)
    S["Z20<-2.5 -> close>SMA20"] = (x.z20 < -2.5, c > x.sma20, 20, None, None)
    S["1-bar drop >3 sigma -> hold 7d"] = (x.dropz1 < -3, c < 0, 7 * d, None, None)
    S["3-day drop >2 sigma -> hold 10d"] = (x.dropz3 < -2, c < 0, 10 * d, None, None)
    S["7-day drop >2 sigma -> hold 20d"] = (x.dropz7 < -2, c < 0, 20 * d, None, None)
    S["close<20d high-3ATR -> close>SMA10"] = (c < x.hi20 - 3 * x.atr, c > x.sma10, 20, None, None)
    S["capitulation: drop>2s & vol>2x -> 10d"] = ((x.dropz1 < -2) & (x.volr > 2), c < 0, 10 * d, None, None)
    S["RSI14 crosses up 30 -> RSI14>55"] = ((x.rsi14 > 30) & (x.rsi14.shift() <= 30), x.rsi14 > 55, 30 * d, None, None)
    S["low IBS<0.15 & down bar -> close>prev high"] = ((x.ibs < 0.15) & (c < c.shift()), c > df.high.shift(), 10, None, None)
    S["Mayer<0.7 (deep below 200d) -> Mayer>1"] = (x.mayer < 0.7, x.mayer > 1.0, 365 * d, None, None)
    # resting buy-limit orders below the last close (a float entry = limit price for the next bar)
    S["limit 2ATR below -> hold 3d"] = (c - 2 * x.atr, c < 0, 3 * d, None, None)
    S["limit 3ATR below -> hold 3d"] = (c - 3 * x.atr, c < 0, 3 * d, None, None)
    S["limit 3ATR below -> close>SMA20"] = (c - 3 * x.atr, c > x.sma20, 10 * d, None, None)
    return S


def run(df, entry, exit_, max_bars, tp, sl):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    limit = entry.dtype.kind == "f"
    entry = entry.to_numpy(float) if limit else entry.fillna(False).to_numpy(bool)
    exit_ = exit_.fillna(False).to_numpy(bool)
    n = len(c)
    eq = np.ones(n)
    cash, units, pos, trades = 1.0, 0.0, False, []
    for i in range(1, n):
        if not pos and (l[i] <= entry[i - 1] if limit else entry[i - 1]):  # NaN limit compares False
            ep, ei, pos = (min(o[i], entry[i - 1]) if limit else o[i] * (1 + SLIP)), i, True
            units, cash = cash * (1 - FEE) / ep, 0.0
        if pos:
            px = None
            if i > ei and (exit_[i - 1] or i - ei >= max_bars):
                px = o[i]
            elif sl and l[i] <= ep * (1 - sl):  # stop checked before TP: conservative when both hit
                px = min(o[i], ep * (1 - sl))
            elif tp and h[i] >= ep * (1 + tp):
                px = max(o[i], ep * (1 + tp))
            if px is not None:
                cash = units * px * (1 - SLIP) * (1 - FEE)
                trades.append((ei, i, cash / (units * ep / (1 - FEE)) - 1))
                pos, units = False, 0.0
        eq[i] = units * c[i] * (1 - FEE) if pos else cash
    return trades, eq


def fwd_baseline(df, regimes):
    """Mean net return of buying at open[t] and selling at open[t+k] on random bars of the same
    period where the same regime filter was on at t-1 (same costs as a trade)."""
    o = df.open.to_numpy()
    is_mask = df.index < SPLIT
    cost = (1 - FEE) ** 2 * (1 - SLIP) / (1 + SLIP)
    cache = {}

    def base(k, oos, f):
        key = (k, oos, f)
        if key not in cache:
            r = o[k:] / o[:-k] * cost - 1
            ok = np.r_[False, regimes[f].to_numpy(bool)[:-1]]
            m = ((~is_mask if oos else is_mask) & ok)[:-k]
            cache[key] = r[m].mean() if m.any() else np.nan
        return cache[key]
    return base


def max_dd(eq):
    return (eq / np.maximum.accumulate(eq) - 1).min()


def evaluate(iv, coins=COINS, filters=("none", "own200", "btc200")):
    bpd = {"1d": 1, "4h": 6, "1h": 24}[iv]
    btc = load("BTCUSDT", iv)
    btc_up = btc.close > btc.close.rolling(200 * bpd).mean()
    rows = []
    for sym in coins:
        df = load(sym, iv)
        x = indicators(df, bpd)
        regime = {"none": pd.Series(True, index=df.index), "own200": df.close > x.sma200,
                  "btc200": btc_up.reindex(df.index).fillna(False)}
        base = fwd_baseline(df, regime)
        for name, (en, ex, mb, tp, sl) in strategies(df, x, bpd).items():
            for f in filters:
                en_f = en.where(regime[f]) if en.dtype.kind == "f" else en & regime[f]
                trades, eq = run(df, en_f, ex, mb, tp, sl)
                for ei, xi, r in trades:
                    oos = df.index[ei] >= SPLIT
                    rows.append(dict(tf=iv, coin=sym[:-4], strat=name, filt=f, period="OOS" if oos else "IS",
                                     ret=r, bars=xi - ei, edge=r - base(xi - ei, oos, f)))
                for per, m in (("IS", df.index < SPLIT), ("OOS", df.index >= SPLIT)):
                    seg = eq[m]
                    if len(seg) > 1:
                        cl = df.close[m]
                        rows.append(dict(tf=iv, coin=sym[:-4], strat=name, filt=f, period=per, eqrow=True,
                                         total=seg[-1] / seg[0] - 1, mdd=max_dd(seg),
                                         bh=cl.iloc[-1] / cl.iloc[0] - 1, bh_mdd=max_dd(cl.to_numpy())))
    return pd.DataFrame(rows)


def summarize(res):
    t = res[res.get("eqrow") != True]
    e = res[res.get("eqrow") == True]
    g = t.groupby(["tf", "strat", "filt", "period"])
    out = pd.DataFrame({
        "trades": g.ret.size(),
        "win%": g.ret.apply(lambda r: (r > 0).mean() * 100),
        "avg%": g.ret.mean() * 100,
        "edge%": g.edge.mean() * 100,
        "t_edge": g.edge.apply(lambda r: r.mean() / r.std() * np.sqrt(len(r)) if len(r) > 2 else np.nan),
        "PF": g.ret.apply(lambda r: r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf),
        "worst%": g.ret.min() * 100,
        "coins+": t.groupby(["tf", "strat", "filt", "period", "coin"]).edge.mean().gt(0).groupby(level=[0, 1, 2, 3]).sum(),
    })
    eg = e.groupby(["tf", "strat", "filt", "period"])
    out["medDD%"] = eg.mdd.median() * 100
    return out.round(2)


if __name__ == "__main__":
    iv = sys.argv[1] if len(sys.argv) > 1 else "1d"
    res = evaluate(iv)
    res.to_pickle(f"{DATA}/res_{iv}.pkl")
    s = summarize(res)
    s.to_csv(f"{DATA}/summary_{iv}.csv")
    wide = s.unstack("period")
    keep = wide[(wide[("trades", "IS")] >= 20) & (wide[("trades", "OOS")] >= 15)]
    pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_columns", 40)
    cols = [(m, p) for p in ("IS", "OOS") for m in ("trades", "win%", "avg%", "edge%", "t_edge", "PF", "coins+", "medDD%")]
    print(keep[cols].sort_values(("edge%", "OOS"), ascending=False).to_string())
