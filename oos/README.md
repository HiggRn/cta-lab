# Out-of-sample (parked)

Out-of-sample work is set aside for now. `main.py` and `uv run pytest` use the in-sample data (`data/raw`, to Dec-2014) only.

- `splice.py`: `check_splice()`, checks that `data/oos` price files continue smoothly from the in-sample files
- `run_oos.py`: runs every strategy across the 2014/2015 join and scores 2015-01 onward. Run from the repo root: `uv run python -m oos.run_oos`
- `test_oos.py`: tests for the join check and for a backtest that runs across the join. Run with `uv run pytest oos`

The data in `data/oos` is a public-data extension to Mar-2024 (Rob Carver's pysystemtrade prices, notes in `data/oos_meta/`). It is noisier than the course data, and EC is a proxy (Fed Funds futures for Eurodollar).

## Known problems before these results can be trusted

1. **Positions in contracts with no new data are never closed.** SS, AP and LB have no data after 2014-12-31. The engine fills an order only on a day the contract has a price bar, so after the last bar even an order to go flat never fills. The 2014 position is held to 2024, earning 0 but paying roll and financing costs. SS is a low-volatility contract, so it got 3-5x of capital in the EWMA strategies (33-41% of gross exposure) and roughly doubled their financing cost.
2. **Strategies keep signals for those contracts.** Signals carry forward and nothing checks `context.active_universe`, so the vol-target scaling keeps counting the dead SS position as risk.

A test fix that closed such positions in the engine and dropped their signals raised the out-of-sample net Sharpe of "Vol target 11%, EWMA risk" from -0.03 to 0.14, and of "Daily EWMA sizing (MOP 2012)" from -0.17 to 0.09. Neither fix is applied: both touch shared core (`engine.py`, `common.py`). No contract ends early in-sample, so the in-sample results are not affected.
