# Pinning + GEX Backtest Data Templates

This folder contains CSV templates for running pinning and gamma exposure (GEX) backtests. Each template lists required columns and sample rows to illustrate formatting.

## Files
- `schema_template_strike_level.csv` – one row per strike with separate columns for calls and puts.
- `schema_template_option_level.csv` – one row per option contract, using `right` to distinguish calls and puts.
- `bars_template.csv` – daily OHLCV bars for the underlying.
- `calendar_template.csv` – trading calendar to disambiguate holidays and weekends.
- `events_template.csv` – optional corporate or macro events that may affect trades.

## Validation Rules
- Open interest must be non‑negative.
- Implied volatility is expressed as a decimal between 0 and 5.
- Bid prices cannot exceed ask prices; no negative quotes.
- All rows for a given `ticker`, `date`, and `expiry` share the same `underlying_price` snapshot.
- Snapshot times should remain consistent across tickers on the same day.
- Filter to the nearest Friday expiry before computing clustering and GEX.
