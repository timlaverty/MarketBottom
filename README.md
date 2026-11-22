<!-- SPDX-FileCopyrightText: 2026 Tim Laverty -->
<!-- SPDX-License-Identifier: MIT -->

# Market-bottom calculation

An exploratory five-indicator market-bottom model with cached-input support and
publication-aware credit timing. The original model is preserved in
`reference/market_bottom_backtest.original.py`, with license comments added.
Dependencies are installed locally under `.venv`.

## License and public distribution

Project-owned source, tests, documentation, and configuration are licensed under
the [MIT License](LICENSE), matching the target repository's existing license.
Downloaded market data, generated reports containing or derived from provider
data, and installed dependencies are outside this license grant. See
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and the publication review in
[PUBLICATION_REVIEW.txt](PUBLICATION_REVIEW.txt).

`data/`, `outputs/`, `.venv/`, Python caches, and `publish/` are excluded from Git.
Keep them out of manual GitHub uploads too: `.gitignore` does not filter files
selected in a browser. The reviewed source archive is generated locally at
`publish/MarketBottom-source.zip`; extract it and upload its contents. It contains
only the reviewed project files and no downloaded data or generated reports.

The public distribution does not include cached inputs. Run online once to
obtain your own inputs, subject to provider terms; offline mode requires a local
cache created previously.

## Running locally

Run from this project in PowerShell:

```powershell
.\.venv\Scripts\python.exe run_market_bottom.py
```

Reproduce the downloaded inputs without internet access:

```powershell
.\.venv\Scripts\python.exe run_market_bottom.py --offline --as-of 2026-11-19
.\.venv\Scripts\python.exe -m unittest -v test_market_bottom.py
```

To recreate the environment, use `python -m venv .venv`, then install
`requirements-lock.txt` for the recorded versions, or `requirements.txt` for
the unpinned dependencies. The recorded runtime is Python 3.14.6, Windows x64.

## Outputs

`outputs/latest/today_report.txt` gives the five individual calculations.
`latest_snapshot.json` records the dates, cutoff, score, interpretation, and
limitations; `today_calculation.csv` gives the indicator table. The same folder
contains `daily_scores.csv`, `signals.csv`, `score_summary.csv`,
`cycle_bottoms.csv`, `benchmark_daily.csv`, and `market_bottom_chart.png`.

Breadth, RSI, and credit-spread rows include a short `reminder` in the JSON/CSV and
text report to explain the indicators.

Locally, `data/prices.csv`, the raw Yahoo responses, and `data/credit_spread.csv`
preserve the downloaded inputs. `data/manifest.json` records source and retrieval
details. These files and all generated outputs stay local and are excluded from
the public distribution.

## Scoring rules

Each rule earns one point:

1. VIX reached at least 35 in 20 trading sessions and has fallen at least 20%
   from that peak.
2. S&P 500 is at least 15% below its 252-session closing high.
3. Adjusted RSP/SPY is above both its 20-session average and its value five
   sessions earlier, **and RSP is above its own 20-session average price**.
   This combines a relative-strength proxy for breadth with an absolute price
   recovery check. Losing less than SPY alone no longer earns the point.
4. Wilder RSI(14) is above 30 after a reading below 30 in the 20-session window.
5. The available high-yield credit spread is below the available value ten
   trading sessions earlier.

Score >=3 is an exploratory bottom-candidate condition; >=4 is a more restrictive
condition. These cutoffs are predefined exploratory settings, not an optimization
or a validated investment recommendation. A score is not a probability.

The daily runner applies this combined breadth rule to both the current score
and historical signals. Recalculated summaries replace prior summaries based
only on the ratio. The extra recovery check can delay signals or miss the first
part of a rebound; its effect on reliability has not been established.
`market_bottom_backtest.py` supplies the legacy evaluation functions; its
standalone calculation retains the earlier breadth rule. Use
`run_market_bottom.py` for the current model and daily report.

## Timing and evaluation details

The runner excludes unfinished US price sessions. It uses only credit
observations dated strictly before each price close, assumes publication by the
next session's close, rejects credit older than seven calendar days, and requires
the ten-session credit comparison to be available. This fixes the supplied
script's use of same-date credit observations before publication. It does not
provide a full vintage/revision audit. A historical `--as-of` applies date
cutoffs to the downloaded data; it cannot reconstruct original publication
vintages or historical ETF adjustment factors.

The full price-session calendar is retained, including dates without enough
credit data to score. Signals require an upward threshold crossing with a
90-trading-session cooldown per threshold. A crossing suppressed during the
cooldown is skipped; the script does not enter later solely because the cooldown
has expired. Entries use the next trading session's close. Today’s score condition
and the historical entry/cooldown rules are reported separately.

The 21/63/126/252-session horizons approximate 1/3/6/12 months. Returns are S&P 500
price returns. The inherited `maxdd_*` columns measure the worst close relative
to the entry price, not a peak-to-trough drawdown after an interim gain. The
runner caps these at zero, correcting positive values in the supplied script.
A false bottom means a close at least 10% below entry within 63 trading sessions.
Incomplete forward windows are excluded, not counted as failures or successes.

FRED's series page states that, starting April 2026, only three years of the ICE
BofA high-yield spread series are supplied. The downloaded credit sample starts
November 20, 2023. Consequently this run does **not** validate the five-factor
model against 2008–09, 2020, or 2022. RSP availability also prevents testing the
same breadth proxy during 2000–02. Full earlier credit history and a compatible
breadth series would be required. No alternative credit series is substituted.

Benchmark columns compare entry-date price returns with unconditional and
same-year/same-drawdown-band controls. They are not a portfolio simulation versus
total-return buy-and-hold. Dividends, transaction costs, cash interest, and taxes
are excluded. Tiny, potentially overlapping signal samples do not establish a
trading advantage. Cycle troughs are ex-post evaluation labels, never score
inputs; only cycles that have recovered the prior peak are confirmed.

Sources: [Yahoo Finance S&P 500](https://finance.yahoo.com/quote/%5EGSPC/history/),
[VIX](https://finance.yahoo.com/quote/%5EVIX/history/),
[RSP](https://finance.yahoo.com/quote/RSP/history/),
[SPY](https://finance.yahoo.com/quote/SPY/history/), and
[FRED credit spread series](https://fred.stlouisfed.org/series/BAMLH0A0HYM2).
