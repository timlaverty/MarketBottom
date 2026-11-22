# SPDX-FileCopyrightText: 2026 Tim Laverty
# SPDX-License-Identifier: MIT

"""Five-indicator market-bottom backtest.

Install: pip install yfinance pandas numpy requests matplotlib
Run: python market_bottom_backtest.py --breadth-csv nyse_breadth.csv
Optional CSV must contain date,new_52w_lows,pct_above_50dma (actual NYSE universe).
Without it, the script explicitly labels and uses RSP/SPY as a breadth proxy.
Outputs in --output-dir: daily_scores.csv, signals.csv, score_summary.csv,
cycle_bottoms.csv, market_bottom_chart.png, benchmark_daily.csv.
Signals use close-of-day data and next-session CLOSE entries. Returns are SPX
price returns (dividends excluded); compare with unconditional entries in the
same sample, and an independent regime-matched control.
"""
import argparse
import io
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt

HORIZONS=(21,63,126,252)
FLAGS=['vix_reversal','large_decline','breadth_recovery','momentum_reversal','credit_stabilization']

def get_prices(start):
    raw=yf.download(['^GSPC','^VIX','RSP','SPY'],start=start,auto_adjust=True,progress=False,threads=False)
    if raw.empty: raise RuntimeError('Yahoo Finance returned no data')
    px=raw['Close'] if isinstance(raw.columns,pd.MultiIndex) else raw
    px=px.rename(columns={'^GSPC':'spx','^VIX':'vix','RSP':'rsp','SPY':'spy'})
    px.index=pd.to_datetime(px.index).tz_localize(None).normalize()
    return px[['spx','vix','rsp','spy']].dropna()

def get_spread():
    url='https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2'
    r=requests.get(url,timeout=35,headers={'User-Agent':'Mozilla/5.0'})
    r.raise_for_status()
    df=pd.read_csv(io.BytesIO(r.content))
    df.columns=['date','hy_spread']
    df['date']=pd.to_datetime(df.date)
    df['hy_spread']=pd.to_numeric(df.hy_spread,errors='coerce')
    return df.set_index('date')['hy_spread']

def rsi(close,n=14):
    delta=close.diff()
    up=delta.clip(lower=0).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    down=(-delta.clip(upper=0)).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    ratio=up/down.replace(0,np.nan)
    return (100-100/(1+ratio)).where(down.ne(0),100)

def compute(start,breadth_csv):
    d=get_prices(start).join(get_spread(),how='left')
    d['hy_spread']=d.hy_spread.ffill(limit=5)
    d['drawdown']=d.spx/d.spx.rolling(252,min_periods=252).max()-1
    d['vix_peak']=d.vix.rolling(20,min_periods=20).max()
    d['vix_reversal']=(d.vix_peak>=35)&(d.vix<=.8*d.vix_peak)
    d['large_decline']=d.drawdown<=-.15
    if breadth_csv:
        b=pd.read_csv(breadth_csv,parse_dates=['date']).set_index('date').sort_index()
        required={'new_52w_lows','pct_above_50dma'}
        if not required.issubset(b.columns): raise ValueError(f'Breadth CSV requires {sorted(required)}')
        if b.index.has_duplicates: raise ValueError('Duplicate dates in breadth CSV')
        b.index=pd.to_datetime(b.index).normalize()
        b=b[['new_52w_lows','pct_above_50dma']].apply(pd.to_numeric,errors='coerce')
        # No forward fill: missing breadth dates must not be treated as observations.
        d=d.join(b)
        d['breadth_recovery']=(d.new_52w_lows<d.new_52w_lows.shift(5)) & (d.pct_above_50dma>d.pct_above_50dma.shift(5))
        d['breadth_available']=d[['new_52w_lows','pct_above_50dma']].notna().all(axis=1)
        breadth_mode='actual NYSE breadth supplied by user'
    else:
        rel=d.rsp/d.spy
        d['breadth_recovery']=(rel>rel.rolling(20).mean())&(rel>rel.shift(5))
        d['breadth_available']=True
        breadth_mode='RSP/SPY relative-strength PROXY (not actual NYSE breadth)'
    d['rsi']=rsi(d.spx)
    d['momentum_reversal']=(d.rsi>30)&(d.rsi.rolling(20).min()<30)
    d['credit_stabilization']=d.hy_spread<d.hy_spread.shift(10)
    d=d.dropna(subset=['drawdown','rsi','hy_spread'])
    d=d.loc[d.breadth_available].copy()
    d['score']=d[FLAGS].sum(axis=1)
    return d,breadth_mode

def forward_table(d):
    """Entry next trading day's close; measure ensuing h trading-day closes."""
    x=d.copy()
    for h in HORIZONS:
        entry=x.spx.shift(-1)
        x[f'ret_{h}d']=x.spx.shift(-(h+1))/entry-1
        # Future min excludes entry, includes exactly h closes after entry.
        future_min=x.spx.shift(-2).iloc[::-1].rolling(h,min_periods=h).min().iloc[::-1]
        x[f'maxdd_{h}d']=future_min/entry-1
    return x

def cycle_bottoms(d,minimum_decline=.15):
    """Ex-post major drawdown troughs: peak-to-trough >=15%, trough until prior peak recovered.
    Unrecovered final cycle is right-censored and omitted from confirmed bottoms.
    """
    p=d.spx.to_numpy()
    peaks=[]
    peak_i=0; trough_i=0; in_dd=False
    for i in range(1,len(p)):
        if p[i]>=p[peak_i]:
            if in_dd and (p[trough_i]/p[peak_i]-1)<=-minimum_decline:
                peaks.append((peak_i,trough_i,i))
            peak_i=i;trough_i=i;in_dd=False
        else:
            in_dd=True
            if p[i]<p[trough_i]:trough_i=i
    rows=[]
    for peak,trough,recovery in peaks:
        rows.append({'peak_date':d.index[peak],'trough_date':d.index[trough],
                     'recovery_date':d.index[recovery], 'peak_spx':p[peak],
                     'trough_spx':p[trough],'peak_to_trough':p[trough]/p[peak]-1,
                     'peak_index':peak,'trough_index':trough,'recovery_index':recovery})
    return pd.DataFrame(rows)

def signal_rows(d,ft,cycles,cooldown):
    rows=[]
    for threshold in (2,3,4,5):
        condition=d.score>=threshold
        crossings=condition & ~condition.shift(1,fill_value=False)
        last=-100000
        for i in np.flatnonzero(crossings.to_numpy()):
            if i-last<cooldown: continue
            last=i
            row={'threshold':threshold,'signal_date':d.index[i], 'score':int(d.score.iloc[i]),
                 'spx_signal':d.spx.iloc[i],'vix':d.vix.iloc[i],
                 'entry_date':d.index[i+1] if i+1<len(d) else pd.NaT}
            for h in HORIZONS:
                row[f'ret_{h}d']=ft[f'ret_{h}d'].iloc[i]
                row[f'maxdd_{h}d']=ft[f'maxdd_{h}d'].iloc[i]
            row['false_bottom_90d']=float(row['maxdd_63d']<=-.10) if pd.notna(row['maxdd_63d']) else np.nan
            # Ex-post trough matching only within a confirmed major cycle; do not
            # call a signal outside one of these cycles a 'miss' of a future bottom.
            row['cycle_trough_date']=pd.NaT
            row['sessions_to_trough']=np.nan
            row['entry_vs_trough_pct']=np.nan
            row['cycle_peak_to_trough']=np.nan
            if not cycles.empty:
                matches=cycles[(cycles.peak_index<=i)&(cycles.recovery_index>=i)]
                if not matches.empty and i+1<len(d):
                    c=matches.iloc[0]
                    row['cycle_trough_date']=c.trough_date
                    row['sessions_to_trough']=int(c.trough_index-i)
                    row['entry_vs_trough_pct']=d.spx.iloc[i+1]/c.trough_spx-1
                    row['cycle_peak_to_trough']=c.peak_to_trough
            rows.append(row)
    return pd.DataFrame(rows)

def summaries(signals,ft):
    # Unconditional benchmark: ALL eligible dates from same complete dataset.
    # Regime-matched benchmark: all dates within same drawdown band and calendar year
    # as signal (excluding its own date), weighted equally per signal.
    out=[]
    bands=pd.cut(ft.drawdown,[-np.inf,-.20,-.15,-.10,-.05,0.01],labels=False)
    for threshold,g in signals.groupby('threshold'):
        s={'score_threshold':threshold,'signals_total':len(g),'false_bottom_rate_90d':g.false_bottom_90d.mean(),
           'matched_confirmed_cycles':g.cycle_trough_date.notna().sum(),
           'mean_sessions_to_cycle_trough':g.sessions_to_trough.mean(),
           'median_entry_above_cycle_trough':g.entry_vs_trough_pct.median()}
        for h in HORIZONS:
            col=f'ret_{h}d'; dd=f'maxdd_{h}d'
            valid=g.dropna(subset=[col])
            x=valid[col]
            s[f'{h}d_n']=len(x)
            s[f'{h}d_mean_return']=x.mean()
            s[f'{h}d_median_return']=x.median()
            s[f'{h}d_positive_rate']=(x>0).mean() if len(x) else np.nan
            s[f'{h}d_mean_maxdd']=valid[dd].mean()
            eligible=ft[col].dropna()
            s[f'{h}d_unconditional_mean_return']=eligible.mean()
            s[f'{h}d_excess_vs_unconditional']=x.mean()-eligible.mean()
            controls=[]
            for dt in valid.signal_date:
                mask=(ft.index.year==dt.year)&(bands==bands.loc[dt])&(ft.index!=dt)
                pool=ft.loc[mask,col].dropna()
                if len(pool):controls.append(pool.mean())
            s[f'{h}d_regime_matched_control_n']=len(controls)
            s[f'{h}d_regime_matched_mean_return']=np.mean(controls) if controls else np.nan
            s[f'{h}d_excess_vs_regime_matched']=x.mean()-np.mean(controls) if len(controls)==len(x) and controls else np.nan
        out.append(s)
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--start',default='2003-06-01')
    ap.add_argument('--breadth-csv',help='CSV with date,new_52w_lows,pct_above_50dma')
    ap.add_argument('--cooldown',type=int,default=90,help='Trading sessions between accepted signals per threshold; 0 disables')
    ap.add_argument('--output-dir',default='.')
    args=ap.parse_args()
    dest=Path(args.output_dir);dest.mkdir(parents=True,exist_ok=True)
    d,mode=compute(args.start,args.breadth_csv)
    ft=forward_table(d)
    cycles=cycle_bottoms(d)
    signals=signal_rows(d,ft,cycles,args.cooldown)
    if signals.empty:raise RuntimeError('No signals found')
    summary=summaries(signals,ft)
    ft.to_csv(dest/'daily_scores.csv')
    signals.to_csv(dest/'signals.csv',index=False)
    summary.to_csv(dest/'score_summary.csv',index=False)
    cycles.drop(columns=['peak_index','trough_index','recovery_index'],errors='ignore').to_csv(dest/'cycle_bottoms.csv',index=False)
    ft[['spx','drawdown']+[f'ret_{h}d' for h in HORIZONS]].to_csv(dest/'benchmark_daily.csv')
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot(d.index,d.spx,label='S&P 500',lw=1)
    for t in (3,4,5):
        ss=signals[signals.threshold==t]
        ax.scatter(ss.signal_date,ss.spx_signal,s=20+15*(t-2),label=f'Score >= {t}',alpha=.7)
    if not cycles.empty:ax.scatter(cycles.trough_date,cycles.trough_spx,marker='v',s=45,label='Confirmed cycle trough',color='black')
    ax.set_yscale('log');ax.set_title('S&P 500 and market-bottom signals (ex-post troughs)')
    ax.legend();fig.tight_layout();fig.savefig(dest/'market_bottom_chart.png',dpi=160)
    print('Sample:',d.index.min().date(),d.index.max().date(),'| Breadth:',mode)
    print('Signals:',len(signals),'| Confirmed recovered drawdown cycles:',len(cycles))
    print('Caution: recent unrecovered bottoms and forward windows are censored.')
    print('Benchmark: all-date unconditional and same-year/same-drawdown-band controls.')
    print(summary.to_string(index=False,float_format=lambda v:f'{v:.3f}'))

if __name__=='__main__':main()
