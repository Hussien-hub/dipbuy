"""Stress-test the ATR buy-limit idea: parameter grid, worse fills, per-year, per-coin, outliers."""
import sys
import numpy as np, pandas as pd
import backtest as bt

iv = sys.argv[1] if len(sys.argv) > 1 else "1d"
d = {"1d": 1, "4h": 6, "1h": 24}[iv]
btc = bt.load("BTCUSDT", iv)
btc_up = btc.close > btc.close.rolling(200 * d).mean()


def trades_for(sym, k, hold_days, filt, through=0.0):
    df = bt.load(sym, iv)
    x = bt.indicators(df, d)
    reg = {"none": pd.Series(True, index=df.index), "own200": df.close > x.sma200,
           "btc200": btc_up.reindex(df.index).fillna(False)}[filt]
    lim = (df.close - k * x.atr).where(reg)
    # through>0: only count a fill if price traded that far below the limit (no touch fills)
    df2 = df.copy()
    df2["low"] = df.low * (1 + through)
    regimes = {"r": reg}
    base = bt.fwd_baseline(df, regimes)
    tr, _ = bt.run(df2, lim, df.close < 0, hold_days * d, None, None)
    return [dict(coin=sym[:-4], date=df.index[e], year=df.index[e].year, ret=r,
                 edge=r - base(xx - e, df.index[e] >= bt.SPLIT, "r")) for e, xx, r in tr]


def stats(t):
    t = pd.DataFrame(t)
    if t.empty:
        return pd.Series(dtype=float)
    return pd.Series({"n": len(t), "win%": (t.ret > 0).mean() * 100, "avg%": t.ret.mean() * 100,
                      "med%": t.ret.median() * 100, "edge%": t.edge.mean() * 100,
                      "t": t.edge.mean() / t.edge.std() * np.sqrt(len(t)),
                      "edge_ex_top5%": t.edge.sort_values().iloc[:-5].mean() * 100 if len(t) > 10 else np.nan})


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(f"##### {iv}: grid (all coins pooled, 2018-2026)")
    rows = {}
    for filt in ("none", "own200"):
        for k in (1.5, 2, 2.5, 3, 3.5, 4):
            for hold in (1, 3, 5, 10):
                t = sum((trades_for(s, k, hold, filt) for s in bt.COINS), [])
                rows[(filt, k, hold)] = stats(t)
    print(pd.DataFrame(rows).T.round(2).to_string())

    for filt, k, hold in (("none", 3, 3), ("own200", 3, 3), ("none", 2, 3)):
        print(f"\n##### {iv}: k={k} hold={hold}d filt={filt}")
        t0 = pd.DataFrame(sum((trades_for(s, k, hold, filt) for s in bt.COINS), []))
        t1 = pd.DataFrame(sum((trades_for(s, k, hold, filt, through=0.005) for s in bt.COINS), []))
        print("touch fills      :", stats(t0).round(2).to_dict())
        print("must trade 0.5%  :", stats(t1).round(2).to_dict())
        print("per coin:\n", t0.groupby("coin").apply(stats, include_groups=False).round(2).to_string())
        print("per year:\n", t0.groupby("year").apply(stats, include_groups=False).round(2).to_string())
        print("worst 5 trades:\n", t0.nsmallest(5, "ret")[["coin", "date", "ret"]].to_string())
