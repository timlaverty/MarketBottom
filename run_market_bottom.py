# SPDX-FileCopyrightText: 2026 Tim Laverty
# SPDX-License-Identifier: MIT

r"""Run the model with cached inputs and publication-aware credit data.

Online:  .venv\Scripts\python.exe run_market_bottom.py
Offline: .venv\Scripts\python.exe run_market_bottom.py --offline
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta, timezone
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import market_bottom_backtest as model

SYMBOLS = {'^GSPC': 'spx', '^VIX': 'vix', 'RSP': 'rsp', 'SPY': 'spy'}
SPREAD_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2'
INDICATOR_REMINDERS = {
    'breadth_recovery': 'Companies beyond the giants must hold up better, and RSP must recover above its own 20-session average.',
    'momentum_reversal': 'Below 30 = strong selling; back above 30 = recovery.',
    'credit_stabilization': 'Extra interest for riskier companies; lower = less lender worry.',
}


def completed_session_cutoff(now):
    """Conservatively exclude an unfinished US session, including a live VIX row."""
    eastern = now.astimezone(ZoneInfo('America/New_York'))
    day = eastern.date()
    if eastern.time() < time(16, 15):
        day -= timedelta(days=1)
    return pd.Timestamp(day)


def refresh_data(start, cutoff, folder):
    folder.mkdir(parents=True, exist_ok=True)
    first = int(pd.Timestamp(start, tz='UTC').timestamp())
    last = int((cutoff.tz_localize('UTC') + pd.Timedelta(days=1)).timestamp())

    def fetch(symbol):
        errors = []
        for host in ('query1.finance.yahoo.com', 'query2.finance.yahoo.com'):
            try:
                url = f'https://{host}/v8/finance/chart/{requests.utils.quote(symbol, safe="")}'
                response = requests.get(url, params={'period1': first, 'period2': last,
                    'interval': '1d'}, headers={'User-Agent': 'Mozilla/5.0'}, timeout=35)
                response.raise_for_status()
                chart = response.json()['chart']['result'][0]
                dates = (pd.to_datetime(chart['timestamp'], unit='s', utc=True)
                         .tz_convert('America/New_York').tz_localize(None).normalize())
                values = chart['indicators'].get('adjclose', [{}])[0].get(
                    'adjclose', chart['indicators']['quote'][0]['close'])
                series = pd.Series(values, index=dates, name=SYMBOLS[symbol]).dropna()
                if series.empty or series.index.has_duplicates or (series <= 0).any():
                    raise ValueError(f'Invalid price data for {symbol}')
                (folder / f'{symbol.replace("^", "")}_yahoo.json').write_text(
                    response.text, encoding='utf-8')
                print(f'{symbol}: {len(series)} rows, latest {series.index.max().date()}', flush=True)
                return series
            except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
                errors.append(str(exc))
        raise RuntimeError(f'Cannot download {symbol}: {errors}')

    with ThreadPoolExecutor(max_workers=4) as pool:
        prices = pd.concat(list(pool.map(fetch, SYMBOLS)), axis=1, sort=True)
    prices.index.name = 'date'
    prices.to_csv(folder / 'prices.csv')
    # FRED's default request headers work with its public CSV endpoint.
    response = requests.get(SPREAD_URL, timeout=35)
    response.raise_for_status()
    if not response.text.startswith('observation_date,'):
        raise ValueError('FRED did not return the expected CSV')
    (folder / 'credit_spread.csv').write_bytes(response.content)
    (folder / 'manifest.json').write_text(json.dumps({
        'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
        'price_source': 'Yahoo Finance daily chart endpoint; adjusted close when supplied',
        'credit_source': SPREAD_URL,
    }, indent=2), encoding='utf-8')


def load_data(folder, cutoff):
    prices = pd.read_csv(folder / 'prices.csv', index_col='date', parse_dates=True).sort_index()
    if prices.index.has_duplicates or not set(SYMBOLS.values()).issubset(prices.columns):
        raise ValueError('Price cache has duplicate dates or missing columns')
    source_dates = {column: str(prices.loc[:cutoff, column].dropna().index.max().date())
                    for column in SYMBOLS.values()}
    prices = prices.loc[:cutoff, list(SYMBOLS.values())].dropna()
    if prices.empty or (prices <= 0).any().any():
        raise ValueError('No valid, completed price sessions')
    spread = pd.read_csv(folder / 'credit_spread.csv', index_col=0, parse_dates=True).iloc[:, 0]
    spread = pd.to_numeric(spread, errors='coerce').dropna().sort_index()
    if spread.empty or spread.index.has_duplicates:
        raise ValueError('Credit cache is empty or has duplicate dates')
    return prices, spread.loc[:cutoff], source_dates


def calculate(prices, spread):
    """Keep the full session calendar so forward horizons remain trading sessions.

    Use only spread observations dated BEFORE a price close. This assumes
    publication by the next session's close, and is not a vintage-data audit.
    """
    d = prices.copy()
    d.index.name = 'date'
    credit = spread.rename('hy_spread').rename_axis('credit_observation_date').reset_index()
    joined = pd.merge_asof(d.reset_index(), credit, left_on='date',
                          right_on='credit_observation_date', direction='backward',
                          allow_exact_matches=False).set_index('date')
    d['credit_observation_date'] = joined.credit_observation_date
    d['hy_spread'] = joined.hy_spread
    d['credit_age_calendar_days'] = (d.index.to_series() - d.credit_observation_date).dt.days
    d['hy_spread'] = d.hy_spread.where(d.credit_age_calendar_days <= 7)
    d['spread_10_sessions_ago'] = d.hy_spread.shift(10)
    d['spread_reference_observation_date'] = d.credit_observation_date.shift(10)
    d['rolling_high_252'] = d.spx.rolling(252, min_periods=252).max()
    d['drawdown'] = d.spx / d.rolling_high_252 - 1
    d['vix_peak'] = d.vix.rolling(20, min_periods=20).max()
    d['vix_reversal'] = (d.vix_peak >= 35) & (d.vix <= .8 * d.vix_peak)
    d['large_decline'] = d.drawdown <= -.15
    d['breadth_ratio'] = d.rsp / d.spy
    d['breadth_mean_20'] = d.breadth_ratio.rolling(20).mean()
    d['breadth_ratio_5_sessions_ago'] = d.breadth_ratio.shift(5)
    d['rsp_mean_20'] = d.rsp.rolling(20, min_periods=20).mean()
    d['breadth_recovery'] = ((d.breadth_ratio > d.breadth_mean_20)
                             & (d.breadth_ratio > d.breadth_ratio_5_sessions_ago)
                             & (d.rsp > d.rsp_mean_20))
    d['rsi'] = model.rsi(d.spx)
    d['rsi_min_20'] = d.rsi.rolling(20).min()
    d['momentum_reversal'] = (d.rsi > 30) & (d.rsi_min_20 < 30)
    d['credit_stabilization'] = d.hy_spread < d.spread_10_sessions_ago
    needed = ['drawdown', 'vix_peak', 'breadth_mean_20', 'breadth_ratio_5_sessions_ago',
              'rsp_mean_20', 'rsi', 'rsi_min_20', 'hy_spread', 'spread_10_sessions_ago']
    d['score_available'] = d[needed].notna().all(axis=1)
    d['score'] = d[model.FLAGS].sum(axis=1).where(d.score_available)
    return d


def latest_details(d, now, cutoff, source_dates, cooldown, signals):
    row = d.iloc[-1]
    available = bool(row.score_available)
    stale = (cutoff - d.index[-1]).days > 4
    score = int(row.score) if available else None
    signal = available and not stale and score >= 3
    indicators = [
        {'indicator': 'VIX reversal', 'flag': 'vix_reversal',
         'calculation': f'VIX {row.vix:.2f}; 20-session peak {row.vix_peak:.2f}; needs peak >=35 and VIX <= {row.vix_peak * .8:.2f}'},
        {'indicator': 'Large decline', 'flag': 'large_decline',
         'calculation': f'S&P 500 {row.spx:,.2f}; 252-session high {row.rolling_high_252:,.2f}; drawdown {row.drawdown:.2%}; needs <= -15%'},
        {'indicator': 'Breadth proxy recovery', 'flag': 'breadth_recovery',
         'calculation': f'Adjusted RSP/SPY {row.breadth_ratio:.6f}; 20-session ratio mean {row.breadth_mean_20:.6f}; 5 sessions ago {row.breadth_ratio_5_sessions_ago:.6f}; RSP ${row.rsp:.2f}; 20-session RSP average ${row.rsp_mean_20:.2f}; needs ratio above both comparisons AND RSP above its own average'},
        {'indicator': 'RSI recovery', 'flag': 'momentum_reversal',
         'calculation': f'RSI(14) {row.rsi:.2f}; 20-session RSI low {row.rsi_min_20:.2f}; needs current >30 and prior low <30'},
        {'indicator': 'Credit stabilization', 'flag': 'credit_stabilization',
         'calculation': f'HY spread {row.hy_spread:.2f}% (observation {row.credit_observation_date.date()}); 10 sessions ago {row.spread_10_sessions_ago:.2f}% (observation {row.spread_reference_observation_date.date()}); needs a decrease'},
    ]
    for indicator in indicators:
        indicator['point'] = int(row[indicator['flag']]) if available else None
        indicator['reminder'] = INDICATOR_REMINDERS.get(indicator['flag'], '')
    latest_signals = {}
    for threshold in (3, 4):
        group = signals[signals.threshold == threshold] if not signals.empty else signals
        latest_signals[str(threshold)] = str(group.signal_date.max().date()) if len(group) else None
    valid = d[d.score_available]
    return {
        'run_at_utc': now.isoformat(),
        'run_at_berlin': now.astimezone(ZoneInfo('Europe/Berlin')).isoformat(),
        'completed_session_cutoff': str(cutoff.date()), 'price_date': str(d.index[-1].date()),
        'source_latest_price_dates': source_dates,
        'credit_observation_date': str(row.credit_observation_date.date()),
        'score_sample_start': str(valid.index.min().date()) if len(valid) else None,
        'score_sample_end': str(valid.index.max().date()) if len(valid) else None,
        'score_available': available, 'prices_stale': stale, 'score': score,
        'buy_threshold': 3, 'stronger_threshold': 4,
        'breadth_rule': 'RSP/SPY above its 20-session mean and five-session-ago value, AND RSP above its own 20-session average',
        'model_buy_condition_met': bool(signal),
        'model_interpretation': ('Insufficient or stale inputs' if not available or stale else
                                'Bottom-candidate buy condition met' if signal else
                                'No market-bottom buy condition'),
        'indicators': indicators, 'cooldown_trading_sessions': cooldown,
        'latest_accepted_threshold_signals': latest_signals,
        'limitations': [
            'A score is a rule count, not a probability that a bottom has occurred.',
            'Threshold 3 is an exploratory cutoff; it is not optimized or validated.',
            'Breadth is the adjusted RSP/SPY proxy, not actual NYSE market breadth.',
            'FRED currently supplies only three years of this credit series; the five-factor backtest cannot cover 2008, 2020, or 2022 with these downloads.',
            'Credit uses the previous available observation date, assuming next-session publication; historical publication vintages and revisions are not audited.',
            'Forward returns are S&P 500 price returns, excluding dividends, transaction costs, taxes, and cash interest.',
            'Columns named maxdd in the supplied code measure the worst loss versus entry, not peak-to-trough drawdown after an interim gain.',
            'The return controls are entry-date comparisons, not an investable strategy versus a total-return buy-and-hold portfolio.',
            'Ex-post cycle troughs are evaluation labels only; unrecovered cycles and incomplete forward windows are censored.',
            'This model does not establish whether an individual investor should buy or sell.'
        ]
    }


def forward_metrics(d):
    ft = model.forward_table(d)
    # Including the entry price makes a path that never falls below entry a
    # zero loss, rather than the positive "drawdown" returned by the old code.
    for horizon in model.HORIZONS:
        ft[f'maxdd_{horizon}d'] = ft[f'maxdd_{horizon}d'].clip(upper=0)
    return ft


def write_outputs(d, source_dates, now, cutoff, dest, cooldown):
    dest.mkdir(parents=True, exist_ok=True)
    ft = forward_metrics(d)
    cycles = model.cycle_bottoms(d)
    signals = model.signal_rows(d, ft, cycles, cooldown)
    if signals.empty:
        signals = pd.DataFrame(columns=['threshold', 'signal_date'])
        summary = pd.DataFrame()
    else:
        summary = model.summaries(signals, ft.loc[d.score_available])
        groups = signals.groupby('threshold').false_bottom_90d
        summary['false_bottom_n_90d'] = summary.score_threshold.map(groups.count())
        summary['false_bottom_count_90d'] = summary.score_threshold.map(groups.sum())
    snapshot = latest_details(d, now, cutoff, source_dates, cooldown, signals)
    ft.to_csv(dest / 'daily_scores.csv')
    signals.to_csv(dest / 'signals.csv', index=False)
    summary.to_csv(dest / 'score_summary.csv', index=False)
    cycles.drop(columns=['peak_index', 'trough_index', 'recovery_index'], errors='ignore').to_csv(
        dest / 'cycle_bottoms.csv', index=False)
    ft.loc[d.score_available, ['spx', 'drawdown'] + [f'ret_{h}d' for h in model.HORIZONS]].to_csv(
        dest / 'benchmark_daily.csv')
    (dest / 'latest_snapshot.json').write_text(json.dumps(snapshot, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(snapshot['indicators']).to_csv(dest / 'today_calculation.csv', index=False)
    text = [f"Market-bottom calculation run {snapshot['run_at_berlin']}",
            f"Prices through {snapshot['price_date']}; credit observation {snapshot['credit_observation_date']}",
            f"Score: {snapshot['score']}/5. {snapshot['model_interpretation']}.", '',
            *[f"{item['point']}/1 {item['indicator']}: {item['calculation']}"
              + (f" | {item['reminder']}" if item['reminder'] else '')
              for item in snapshot['indicators']],
            '', f"Five-factor sample: {snapshot['score_sample_start']} to {snapshot['score_sample_end']}",
            'Limitations:', *snapshot['limitations']]
    (dest / 'today_report.txt').write_text('\n'.join(text) + '\n', encoding='utf-8')
    recent = d.tail(504)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1, 1]})
    axes[0].plot(recent.index, recent.spx, color='#244d78', label='S&P 500 close')
    if not signals.empty:
        buys = signals[(signals.threshold == 3) & (signals.signal_date >= recent.index.min())]
        axes[0].scatter(buys.signal_date, buys.spx_signal, marker='^', color='#16856a', label='Accepted score >=3 signals')
    axes[0].set_title(f"Market-bottom model: {snapshot['score']}/5 | Prices {snapshot['price_date']}")
    axes[0].legend(loc='upper left')
    axes[1].step(recent.index, recent.score, where='post', color='#16856a')
    axes[1].axhline(3, color='#b06622', linestyle='--', label='Exploratory buy cutoff')
    axes[1].set(ylim=(-.2, 5.2), yticks=range(6), ylabel='Score / 5')
    axes[1].legend(loc='upper left')
    axes[2].plot(recent.index, recent.drawdown * 100, color='#9e4146')
    axes[2].axhline(-15, color='#b06622', linestyle='--')
    axes[2].set_ylabel('252-session\ndrawdown (%)')
    for ax in axes:
        ax.grid(alpha=.18)
    fig.text(.08, .015, 'RSP/SPY breadth proxy; credit lagged by observation date. Historical associations are not a validated trading edge.', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(dest / 'market_bottom_chart.png', dpi=160)
    plt.close(fig)
    print('\n'.join(text[:10]), flush=True)
    if not summary.empty:
        cols = ['score_threshold', 'signals_total', 'false_bottom_rate_90d',
                '21d_n', '21d_positive_rate', '63d_n', '63d_positive_rate',
                '126d_n', '126d_positive_rate', '126d_mean_return']
        print(summary[cols].to_string(index=False), flush=True)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use the recorded CSV inputs without network access')
    parser.add_argument('--start', default='2003-06-01')
    parser.add_argument('--as-of', help='Maximum completed price date (YYYY-MM-DD); never permits an unfinished session')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/latest'))
    parser.add_argument('--cooldown', type=int, default=90)
    args = parser.parse_args()
    if args.cooldown < 0:
        parser.error('--cooldown must be nonnegative')
    now = datetime.now(timezone.utc)
    cutoff = completed_session_cutoff(now)
    if args.as_of:
        cutoff = min(cutoff, pd.Timestamp(args.as_of))
    if not args.offline:
        refresh_data(args.start, cutoff, args.data_dir)
    prices, spread, source_dates = load_data(args.data_dir, cutoff)
    d = calculate(prices.loc[args.start:], spread)
    write_outputs(d, source_dates, now, cutoff, args.output_dir, args.cooldown)


if __name__ == '__main__':
    main()
