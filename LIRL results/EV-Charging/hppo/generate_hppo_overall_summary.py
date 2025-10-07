#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 HPPO multi_run_overall_summary_* CSV (仿 LIRL/PDQN)"""
from __future__ import annotations
import os, glob
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REQUIRED_COLS = [
    'Run','Episode','Total_Arrivals','Vehicles_Charged','Vehicles_Left_Uncharged',
    'Charging_Success_Rate_%','Cumulative_Revenue','Energy_Delivered_kWh',
    'Average_Power_kW','Station_Utilization_%','Constraint_Violations','Violation_Rate_%'
]
METRIC_UNITS = {
    'Total Arrivals per Run':'vehicles',
    'Total Charged per Run':'vehicles',
    'Total Left Uncharged per Run':'vehicles',
    'Success Rate per Run':'%',
    'Average Revenue per Run':'currency',
    'Average Energy per Run':'kWh',
    'Average Power per Run':'kW',
    'Average Station Utilization per Run':'%',
    'Total Violations per Run':'violations',
    'Violation Rate per Run':'%',
}

def latest_all_episodes()->str:
    files = glob.glob(os.path.join(BASE_DIR,'*all_episodes*.csv'))
    if not files:
        raise FileNotFoundError('未找到 all_episodes CSV')
    return sorted(files)[-1]

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f'缺失列: {c}')
    df = df.sort_values(['Run','Episode'])
    last = df.groupby('Run').tail(1)
    g = df.groupby('Run')
    arrivals = g['Total_Arrivals'].sum()
    charged = g['Vehicles_Charged'].sum()
    left = g['Vehicles_Left_Uncharged'].sum()
    success = (charged/arrivals*100)
    avg_power = g['Average_Power_kW'].mean()
    avg_util = g['Station_Utilization_%'].mean()
    avg_viol_rate = g['Violation_Rate_%'].mean()
    viol_total = g['Constraint_Violations'].sum()
    final_rev = last.set_index('Run')['Cumulative_Revenue']
    final_energy = last.set_index('Run')['Energy_Delivered_kWh']
    per_run = {
        'Total Arrivals per Run': arrivals,
        'Total Charged per Run': charged,
        'Total Left Uncharged per Run': left,
        'Success Rate per Run': success,
        'Average Revenue per Run': final_rev,
        'Average Energy per Run': final_energy,
        'Average Power per Run': avg_power,
        'Average Station Utilization per Run': avg_util,
        'Total Violations per Run': viol_total,
        'Violation Rate per Run': avg_viol_rate,
    }
    return pd.DataFrame(per_run).reset_index(drop=True)

def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for col in per_run.columns:
        s=per_run[col]
        rows.append({'Metric':col,'Mean':s.mean(),'Std':s.std(ddof=1),'Min':s.min(),'Max':s.max(),'Unit':METRIC_UNITS.get(col,'')})
    return pd.DataFrame(rows)

def main():
    path = latest_all_episodes()
    print('使用文件:', path)
    # 尝试多种方式读取，规避某些编码/路径问题
    encodings_try = ['utf-8-sig','utf-8','gbk','latin1']
    df = None
    for enc in encodings_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        # 二进制手动尝试
        import io
        raw = open(path,'rb').read()
        for enc in encodings_try:
            try:
                text = raw.decode(enc)
                df = pd.read_csv(io.StringIO(text))
                break
            except Exception:
                continue
    if df is None:
        raise RuntimeError('无法读取 CSV 文件，尝试的编码: ' + ','.join(encodings_try))
    per_run = aggregate(df)
    summary = summarize(per_run)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(BASE_DIR,f'hppo_multi_run_overall_summary_{ts}.csv')
    summary.to_csv(out,index=False,encoding='utf-8-sig')
    print('已生成:', out)
    print(summary)

if __name__=='__main__':
    main()
