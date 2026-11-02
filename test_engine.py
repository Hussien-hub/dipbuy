"""Self-check of the trade engine on a hand-made series."""
import numpy as np, pandas as pd
import backtest as bt

idx = pd.date_range("2022-01-01", periods=6, freq="D")
df = pd.DataFrame({"open": [100, 100, 90, 95, 110, 120], "high": [101, 101, 96, 112, 121, 121],
                   "low": [99, 89, 88, 94, 109, 119], "close": [100, 90, 95, 110, 120, 120]}, index=idx)
entry = pd.Series([False, True, False, False, False, False], index=idx)   # signal on bar 1 close
exit_ = pd.Series([False, False, False, True, False, False], index=idx)   # exit signal on bar 3 close
trades, eq = bt.run(df, entry, exit_, 10, None, None)
(ei, xi, r), = trades
assert (ei, xi) == (2, 4), trades                                           # fill next opens: 90 -> 110
expect = 110 * (1 - bt.SLIP) * (1 - bt.FEE) ** 2 / (90 * (1 + bt.SLIP)) - 1
assert abs(r - expect) < 1e-12 and abs(eq[-1] - (1 + expect)) < 1e-12, (r, expect)
# stop loss gap: entry 90, 5% stop = 85.5, bar 2 low 88 not hit; time stop after 1 bar exits at open[3]
trades, _ = bt.run(df, entry, exit_ & False, 1, None, 0.05)
assert trades[0][:2] == (2, 3), trades
print("engine ok", round(r * 100, 3), "%")
# limit order: limit 92 set on bar 1 close, bar 2 low 88 <= 92 and open 90 < 92 -> fill at open 90
lim = pd.Series([np.nan, 92.0, np.nan, np.nan, np.nan, np.nan], index=idx)
trades, _ = bt.run(df, lim, exit_, 10, None, None)
assert trades[0][:2] == (2, 4), trades
lim2 = pd.Series([np.nan, 87.0, np.nan, np.nan, np.nan, np.nan], index=idx)   # never reached (low 88)
assert bt.run(df, lim2, exit_, 10, None, None)[0] == []
print("limit ok")
