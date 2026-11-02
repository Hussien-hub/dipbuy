"""Daily data: (1) forward returns after a dip, by how deep the dip is; (2) DCA variants that use dips.
Entry at next day's open, so nothing uses information not known at the signal close."""
import numpy as np, pandas as pd
from backtest import load, rsi, COINS, SPLIT, FEE, SLIP

H = [7, 30, 90, 180]


def frame(sym):
    df = load(sym, "1d")
    c = df.close
    x = pd.DataFrame(index=df.index)
    x["coin"] = sym[:-4]
    x["period"] = np.where(df.index < SPLIT, "IS", "OOS")
    x["d1"] = c.pct_change()
    x["mayer"] = c / c.rolling(200).mean()
    x["dd365"] = c / df.high.rolling(365, min_periods=200).max() - 1
    x["rsi14"] = rsi(c, 14)
    entry = df.open.shift(-1) * (1 + SLIP) / (1 - FEE)
    for h in H:
        x[f"f{h}"] = c.shift(-h) / entry - 1
    return x


def table(x, col, bins, labels):
    x = x.assign(b=pd.cut(x[col], bins, labels=labels))
    g = x.groupby(["b"], observed=True)
    out = pd.DataFrame({"days": g.size(), "coins": g.coin.nunique()})
    for h in H:
        out[f"med{h}d%"] = g[f"f{h}"].median() * 100
        out[f"up{h}d%"] = g[f"f{h}"].apply(lambda r: (r.dropna() > 0).mean() * 100)
    return out.round(1)


def dca(df_sym, weight, budget=100.0):
    """Every Monday add `budget` to cash; spend budget*weight (capped by cash). Returns $ value / $ deposited."""
    c, o = df_sym.close, df_sym.open.shift(-1)
    cash = units = deposited = 0.0
    for t in df_sym.index[df_sym.index.dayofweek == 0][:-1]:
        cash += budget
        deposited += budget
        spend = min(cash, budget * weight(t))
        if spend > 0:
            units += spend * (1 - FEE) / (o[t] * (1 + SLIP))
            cash -= spend
    return (units * c.iloc[-1] + cash) / deposited, cash / deposited


if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_columns", 30)
    X = pd.concat([frame(s) for s in COINS]).dropna(subset=["mayer"])

    print("=== ALL DAYS (baseline) ===")
    print(table(X, "d1", [-1, 9], ["any day"]).to_string())

    for per in ("IS", "OOS", None):
        sub = X if per is None else X[X.period == per]
        tag = per or "ALL 2018-2026"
        print(f"\n=== {tag}: by size of today's drop (1-day close-to-close) ===")
        print(table(sub, "d1", [-1, -0.10, -0.07, -0.05, -0.03, 0, 9],
                    ["<-10%", "-10..-7%", "-7..-5%", "-5..-3%", "-3..0%", "up day"]).to_string())
        print(f"\n=== {tag}: by Mayer multiple (price / 200-day SMA) ===")
        print(table(sub, "mayer", [0, 0.6, 0.8, 1.0, 1.2, 1.5, 99],
                    ["<0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", ">1.5"]).to_string())
        print(f"\n=== {tag}: by drawdown from 365-day high ===")
        print(table(sub, "dd365", [-1, -0.8, -0.6, -0.4, -0.2, 0.01],
                    ["<-80%", "-80..-60%", "-60..-40%", "-40..-20%", "-20..0%"]).to_string())

    print("\n=== Drop-day x trend: drop < -5% today, split by Mayer ===")
    big = X[X.d1 < -0.05]
    print(table(big, "mayer", [0, 0.8, 1.0, 1.2, 99], ["<0.8", "0.8-1.0", "1.0-1.2", ">1.2"]).to_string())

    print("\n=== Per coin: 90d median fwd return by Mayer bucket ===")
    X["mb"] = pd.cut(X.mayer, [0, 0.6, 0.8, 1.0, 1.2, 1.5, 99], labels=["<0.6", "0.6-0.8", "0.8-1.0", "1.0-1.2", "1.2-1.5", ">1.5"])
    print((X.groupby(["coin", "mb"], observed=True).f90.median().unstack() * 100).round(0).to_string())

    print("\n=== DCA: $ value per $ deposited at 2026-11-01 (higher = better); idle cash % in () ===")
    rules = {
        "plain weekly DCA": lambda m, d: 1.0,
        "Mayer-weighted (3x<0.8, 2x<1, 1x<1.5, 0.5x)": lambda m, d: 3 if m < 0.8 else 2 if m < 1 else 1 if m < 1.5 else 0.5,
        "only buy below 200d SMA (bank cash otherwise)": lambda m, d: 99 if m < 1 else 0,
        "only buy 30%+ under 365d high (bank otherwise)": lambda m, d: 99 if d < -0.3 else 0,
    }
    for start in ("2018-01-01", "2022-01-01", "2024-01-01"):
        rows = {}
        for sym in COINS:
            df = load(sym, "1d")
            df = df[df.index >= max(pd.Timestamp(start), df.index[0] + pd.Timedelta(days=200))]
            full = frame(sym)
            row = {}
            for name, rule in rules.items():
                m, d = full.mayer, full.dd365
                v, idle = dca(df, lambda t: rule(m[t], d[t]))
                row[name] = f"{v:.2f} ({idle * 100:.0f}%)"
            rows[sym[:-4]] = row
        print(f"\n-- start {start} --")
        print(pd.DataFrame(rows).to_string())

    print("\n=== Where each coin is now ===")
    now = pd.concat([frame(s).tail(1)[["coin", "d1", "mayer", "dd365", "rsi14"]] for s in COINS])
    print(now.round(3).to_string())
