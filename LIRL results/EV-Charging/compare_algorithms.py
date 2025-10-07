#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
比较 lirl, hppo, pdqn 三个算法的性能：
1. 读取各自的 scores npy（假设形状: (num_runs, num_episodes) 或 (num_runs,)）并计算均值等统计（可扩展）。
2. 读取各自的 overall summary CSV：
   - lirl: multi_run_overall_summary_*.csv
   - pdqn: pdqn_multi_run_overall_summary_*.csv
   - hppo: 没有 overall summary, 用 vehicle_flow_summary 或直接从 all_episodes 聚合生成对应指标。
3. 统一选择一组核心指标绘制雷达图：
   ['Success Rate per Run','Average Revenue per Run','Average Energy per Run','Average Power per Run','Average Station Utilization per Run','Violation Rate per Run']
4. 输出：
   - 控制台打印对比表（均值）
   - 保存 radar 图 compare_radar.png
   - 保存聚合后的对比 CSV compare_metrics.csv

使用：
python compare_algorithms.py
"""
from __future__ import annotations
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ALG_DIRS = {
    'lirl': 'lirl',
    'hppo': 'hppo',
    'pdqn': 'pdqn'
}
# 目标雷达图指标（Mean 列）
RADAR_METRICS = [
    'Success Rate per Run',
    'Average Revenue per Run',
    'Average Energy per Run',
    'Average Power per Run',
    'Average Station Utilization per Run',
    'Violation Rate per Run'
]

def find_file(pattern: str, directory: str) -> str | None:
    paths = glob.glob(os.path.join(directory, pattern))
    return paths[0] if paths else None

# -------- 读取 LIRL / PDQN Overall Summary ---------

def load_overall_summary(alg_name: str, folder: str) -> pd.DataFrame | None:
    # 这类 CSV 第一行包含 Metric 结构
    csv_files = glob.glob(os.path.join(folder, '*overall_summary*.csv'))
    if not csv_files:
        return None
    # 选最新或第一个
    csv_path = os.path.abspath(sorted(csv_files)[-1])
    original_path = csv_path
        # 处理 Windows 长路径（当前无需强制添加前缀，下面代码仅作参考）
        # if os.name == 'nt' and not csv_path.startswith('\\\\?\\'):
        #     csv_path = '\\?\\' + csv_path  # type: ignore
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e1:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e2:
            try:
                # 最后尝试二进制读取再解码
                with open(csv_path, 'rb') as f:
                    import io
                    raw = f.read()
                    for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin1']:
                        try:
                            text = raw.decode(enc)
                            df = pd.read_csv(io.StringIO(text))  # type: ignore
                            break
                        except Exception:
                            df = None  # type: ignore
                    if df is None:  # type: ignore
                        raise RuntimeError(f"无法解析 {original_path} 使用多种编码, 原始错误: {e1}; {e2}")
            except Exception as e3:
                raise RuntimeError(f"读取 summary 失败: {original_path}: {e1}; {e2}; {e3}")
    # 期望列: Metric, Mean, Std, Min, Max, Unit
    if 'Metric' not in df.columns:
        return None
    return df

# -------- 读取 HPPO 并聚合生成与其它一致的汇总 ---------

def aggregate_hppo(folder: str) -> pd.DataFrame | None:
    # 查找 vehicle_flow_summary (per-run 粒度基础指标)
    vf_files = glob.glob(os.path.join(folder, '*vehicle_flow_summary*.csv'))
    if not vf_files:
        return None
    vf_path = sorted(vf_files)[-1]
    vf_df = None
    for enc in ['utf-8-sig','utf-8','gbk','latin1']:
        try:
            vf_df = pd.read_csv(vf_path, encoding=enc)
            break
        except Exception:
            continue
    if vf_df is None:
        import io
        raw = open(vf_path,'rb').read()
        for enc in ['utf-8-sig','utf-8','gbk','latin1']:
            try:
                text = raw.decode(enc)
                vf_df = pd.read_csv(io.StringIO(text))
                break
            except Exception:
                continue
    if vf_df is None:
        raise RuntimeError(f"无法读取 HPPO vehicle_flow_summary: {vf_path}")

    rename_map = {
        'Total_Arrivals': 'Total Arrivals per Run',
        'Charged': 'Total Charged per Run',
        'Left_Uncharged': 'Total Left Uncharged per Run',
        'Success_Rate_%': 'Success Rate per Run',
        'Avg_Revenue': 'Average Revenue per Run',
        'Avg_Energy_kWh': 'Average Energy per Run'
    }
    for c in rename_map.keys():
        if c not in vf_df.columns:
            raise ValueError(f"HPPO 缺少列 {c} 于 {vf_path}")
    vf_df = vf_df.rename(columns=rename_map)

    # 尝试从 all_episodes 提取 Power / Utilization / Violation 统计
    avg_power = util = viol_rate = None
    ae_files = glob.glob(os.path.join(folder, '*all_episodes*.csv'))
    if ae_files:
        ae_path = sorted(ae_files)[-1]
        ae_df = None
        for enc in ['utf-8-sig','utf-8','gbk','latin1']:
            try:
                ae_df = pd.read_csv(ae_path, encoding=enc)
                break
            except Exception:
                continue
        if ae_df is None:
            import io
            raw = open(ae_path,'rb').read()
            for enc in ['utf-8-sig','utf-8','gbk','latin1']:
                try:
                    text = raw.decode(enc)
                    ae_df = pd.read_csv(io.StringIO(text))
                    break
                except Exception:
                    continue
        if ae_df is not None and 'Run' in ae_df.columns:
            grp = ae_df.groupby('Run')
            if 'Average_Power_kW' in ae_df.columns:
                avg_power = grp['Average_Power_kW'].mean().mean()  # 先 run 内均值，再对 runs 取均值
            if 'Station_Utilization_%' in ae_df.columns:
                util = grp['Station_Utilization_%'].mean().mean()
            if 'Violation_Rate_%' in ae_df.columns:
                viol_rate = grp['Violation_Rate_%'].mean().mean()

    # 构建 summary
    rows: List[dict] = []
    def add_stat(metric_name: str, series: pd.Series, unit: str):
        rows.append({
            'Metric': metric_name,
            'Mean': series.mean(),
            'Std': series.std(),
            'Min': series.min(),
            'Max': series.max(),
            'Unit': unit
        })

    add_stat('Success Rate per Run', vf_df['Success Rate per Run'], '%')
    add_stat('Average Revenue per Run', vf_df['Average Revenue per Run'], '')
    add_stat('Average Energy per Run', vf_df['Average Energy per Run'], 'kWh')

    # 如果有衍生指标，直接按汇总值填入（Std 为 0）
    if avg_power is not None:
        rows.append({'Metric': 'Average Power per Run','Mean': avg_power,'Std': 0,'Min': avg_power,'Max': avg_power,'Unit': 'kW'})
    if util is not None:
        rows.append({'Metric': 'Average Station Utilization per Run','Mean': util,'Std': 0,'Min': util,'Max': util,'Unit': '%'})
    if viol_rate is not None:
        rows.append({'Metric': 'Violation Rate per Run','Mean': viol_rate,'Std': 0,'Min': viol_rate,'Max': viol_rate,'Unit': '%'})

    return pd.DataFrame(rows)

# -------- 读取 scores.npy 统计 (optional) ---------

def load_scores_stats(alg_name: str, folder: str) -> Dict[str, float]:
    pattern_parts = {
        'lirl': 'ddpg_lirl_pi_all_scores_*.npy',
        'hppo': 'hppo_scores_*.npy',
        'pdqn': 'pdqn_scores_*.npy'
    }
    pattern = pattern_parts[alg_name]
    paths = glob.glob(os.path.join(folder, pattern))
    if not paths:
        return {}
    path = sorted(paths)[-1]
    arr = np.load(path, allow_pickle=True)
    # 统计（假设 arr 为 2D (runs, episodes) 或 1D）
    arr_flat = arr.reshape(-1)
    return {
        'scores_mean': float(arr_flat.mean()),
        'scores_std': float(arr_flat.std()),
        'scores_min': float(arr_flat.min()),
        'scores_max': float(arr_flat.max()),
        'scores_file': os.path.basename(path)
    }

# -------- 主流程 ---------

def main():
    summary_map: Dict[str, pd.DataFrame] = {}
    for alg, sub in ALG_DIRS.items():
        folder = os.path.join(BASE_DIR, sub)
        if alg == 'hppo':
            # 优先查找已经生成的 overall summary
            existing = glob.glob(os.path.join(folder, '*overall_summary*.csv'))
            existing = [p for p in existing if 'vehicle_flow' not in os.path.basename(p)]
            if existing:
                try:
                    df = pd.read_csv(sorted(existing)[-1])
                except Exception:
                    df = None
                if df is None or 'Metric' not in df.columns:
                    df = aggregate_hppo(folder)
            else:
                df = aggregate_hppo(folder)
        else:
            df = load_overall_summary(alg, folder)
        if df is None:
            print(f"[WARN] 未找到 {alg} 的汇总文件")
            continue
        summary_map[alg] = df

    # 构建对比表（仅保留目标指标）
    compare_rows = []
    for alg, df in summary_map.items():
        for metric in RADAR_METRICS:
            row = df[df['Metric'] == metric]
            if not row.empty:
                compare_rows.append({'Algorithm': alg, 'Metric': metric, 'Mean': float(row['Mean'].values[0])})
            else:
                compare_rows.append({'Algorithm': alg, 'Metric': metric, 'Mean': np.nan})
    compare_df = pd.pivot_table(pd.DataFrame(compare_rows), index='Metric', columns='Algorithm', values='Mean')

    # 保存对比 CSV
    out_csv = os.path.join(BASE_DIR, 'compare_metrics.csv')
    compare_df.to_csv(out_csv, encoding='utf-8-sig')
    print(f"已保存对比指标: {out_csv}")

    # 打印表格
    print('\n=== 指标均值对比 ===')
    print(compare_df)

    # 雷达图：对缺失值用列平均或 0 填补，这里用最小-最大归一化前先填补为整体均值
    filled = compare_df.copy()
    for metric in filled.index:
        if filled.loc[metric].isna().any():
            filled.loc[metric] = filled.loc[metric].fillna(filled.loc[metric].mean())

    # 归一化：使用 value / max(metric) 作为尺度；Violation Rate 仍反向（越低越好）
    norm = filled.copy()
    for metric in norm.index:
        series = norm.loc[metric]
        mx = series.max()
        if mx == 0:
            norm.loc[metric] = 0.0
            continue
        ratio = series / mx
        if metric == 'Violation Rate per Run':
            # 反向：低的更好 -> 1 - (value/max)
            ratio = 1 - ratio
        norm.loc[metric] = ratio

    # Radar plot
    labels = norm.index.tolist()
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    colors = {'lirl':'#1f77b4','hppo':'#ff7f0e','pdqn':'#2ca02c'}
    for alg in compare_df.columns:
        if alg not in norm.columns:
            continue
        values = norm[alg].tolist()
        values += values[:1]
        # 若全部为 0（算法在所有指标上皆为最差，导致雷达图塌缩到中心），给一个很小的可视化抬升
        if all(v == 0 for v in values[:-1]):
            values = [0.02]*(len(values)-1)
            values += values[:1]
            label = alg.upper() + ' (all-min)'
        else:
            label = alg.upper()
        ax.plot(angles, values, label=label, linewidth=2, marker='o', markersize=4, color=colors.get(alg,'gray'))
        ax.fill(angles, values, alpha=0.15, color=colors.get(alg,'gray'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties='SimHei', fontsize=10)
    ax.set_yticklabels([])
    ax.set_title('算法性能雷达图 (归一化后)', fontproperties='SimHei')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    out_png = os.path.join(BASE_DIR, 'compare_radar.png')
    plt.savefig(out_png, dpi=200)
    print(f"已保存雷达图: {out_png}")

    # ====== 额外：原始数值比例雷达图（每指标独立 min-max 缩放，仅用于展示原始相对差异，不反转违规率） ======
    raw_df = compare_df.copy()
    # 填补缺失
    for metric in raw_df.index:
        if raw_df.loc[metric].isna().any():
            raw_df.loc[metric] = raw_df.loc[metric].fillna(raw_df.loc[metric].mean())
    # 逐指标做 value / max(metric)（不反向）
    scaled = raw_df.copy()
    for metric in scaled.index:
        series = scaled.loc[metric]
        mx = series.max()
        if mx > 0:
            scaled.loc[metric] = series / mx
        else:
            scaled.loc[metric] = 0.0
    labels_raw = scaled.index.tolist()
    angles_raw = np.linspace(0, 2 * np.pi, len(labels_raw), endpoint=False).tolist()
    angles_raw += angles_raw[:1]
    fig2 = plt.figure(figsize=(8,8))
    ax2 = plt.subplot(111, polar=True)
    for alg in scaled.columns:
        vals = scaled[alg].tolist(); vals += vals[:1]
        ax2.plot(angles_raw, vals, label=alg.upper(), linewidth=2, marker='o')
        ax2.fill(angles_raw, vals, alpha=0.12)
    ax2.set_xticks(angles_raw[:-1])
    ax2.set_xticklabels(labels_raw, fontproperties='SimHei', fontsize=10)
    ax2.set_yticklabels([])
    ax2.set_title('算法性能雷达图 (原始数值按各指标独立缩放)', fontproperties='SimHei')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.25,1.1))
    plt.tight_layout()
    out_png_raw = os.path.join(BASE_DIR, 'compare_radar_raw.png')
    plt.savefig(out_png_raw, dpi=200)
    print(f"已保存原始数值雷达图: {out_png_raw}")

    # ====== 额外：Soft-Floor 雷达图（避免最差算法完全贴近中心） ======
    soft_floor = 0.15  # 最低可视化半径
    soft_norm = norm.copy()
    for metric in soft_norm.index:
        series = soft_norm.loc[metric]
        # 仅提升为 0 的点
        series = series.apply(lambda v: soft_floor if v == 0 else v)
        # 若所有值仍相等（提升后），保持不动
        soft_norm.loc[metric] = series
    labels_sf = soft_norm.index.tolist()
    angles_sf = np.linspace(0, 2 * np.pi, len(labels_sf), endpoint=False).tolist()
    angles_sf += angles_sf[:1]
    fig3 = plt.figure(figsize=(8,8))
    ax3 = plt.subplot(111, polar=True)
    for alg in soft_norm.columns:
        vals = soft_norm[alg].tolist(); vals += vals[:1]
        ax3.plot(angles_sf, vals, label=alg.upper(), linewidth=2, marker='o')
        ax3.fill(angles_sf, vals, alpha=0.12)
    ax3.set_xticks(angles_sf[:-1])
    ax3.set_xticklabels(labels_sf, fontproperties='SimHei', fontsize=10)
    ax3.set_yticklabels([])
    ax3.set_title(f'算法性能雷达图 (Soft-Floor={soft_floor})', fontproperties='SimHei')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.25,1.1))
    plt.tight_layout()
    out_png_soft = os.path.join(BASE_DIR, 'compare_radar_soft.png')
    plt.savefig(out_png_soft, dpi=200)
    print(f"已保存 Soft-Floor 雷达图: {out_png_soft}")

    # ====== 额外：分组柱状图（原始均值绝对值对比） ======
    bar_df = compare_df.copy()
    # 绘制时对指标顺序可重新排序（保持当前）
    fig4, ax4 = plt.subplots(figsize=(10,6))
    metrics_order = bar_df.index.tolist()
    algs = bar_df.columns.tolist()
    x = np.arange(len(metrics_order))
    width = 0.8 / len(algs)
    # 为显著性检验准备 per-run 数据（仅能使用我们能解析的来源）
    # 策略：
    #  - lirl & pdqn: 使用各自 overall summary 无法获得 per-run -> 若存在 multi_run_overall_summary 可以不做 per-run; 暂时只能对 HPPO + (不可比) 跳过
    #  - 如果未来添加 per-run CSV，可扩展此逻辑。
    # 现在：尝试从以下来源构造 per-run 指标：
    #   * hppo: vehicle_flow_summary (已有 per-run 行)
    #   * lirl/pdqn: 若存在 multi_run_overall_summary_* 无 per-run, 退出显著性比较; 仅生成空p值表
    per_run_metrics = ['Success Rate per Run','Average Revenue per Run','Average Energy per Run','Average Power per Run','Average Station Utilization per Run','Violation Rate per Run']
    per_run_data = {}
    # HPPO per-run
    hppo_vf = glob.glob(os.path.join(BASE_DIR,'hppo','*vehicle_flow_summary*.csv'))
    if hppo_vf:
        vf_path = sorted(hppo_vf)[-1]
        hppo_df = None
        for enc in ['utf-8-sig','utf-8','gbk','latin1']:
            try:
                hppo_df = pd.read_csv(vf_path, encoding=enc)
                break
            except Exception:
                continue
        if hppo_df is None:
            import io
            raw = open(vf_path,'rb').read()
            for enc in ['utf-8-sig','utf-8','gbk','latin1']:
                try:
                    text = raw.decode(enc)
                    hppo_df = pd.read_csv(io.StringIO(text))
                    break
                except Exception:
                    continue
        if hppo_df is None:
            print(f"[WARN] 无法读取 HPPO vehicle_flow_summary 用于显著性检验: {vf_path}")
        else:
            # rename to match
            rename_map_local = {
                'Total_Arrivals': 'Total Arrivals per Run',
                'Charged': 'Total Charged per Run',
                'Left_Uncharged': 'Total Left Uncharged per Run',
                'Success_Rate_%': 'Success Rate per Run',
                'Avg_Revenue': 'Average Revenue per Run',
                'Avg_Energy_kWh': 'Average Energy per Run'
            }
            hppo_df = hppo_df.rename(columns=rename_map_local)
            # Need power, util, violation rate from all_episodes aggregated per run
            hppo_all_ep = glob.glob(os.path.join(BASE_DIR,'hppo','*all_episodes*.csv'))
            hppo_extra = None
            if hppo_all_ep:
                ep_path = sorted(hppo_all_ep)[-1]
                ep_df = None
                for enc in ['utf-8-sig','utf-8','gbk','latin1']:
                    try:
                        ep_df = pd.read_csv(ep_path, encoding=enc)
                        break
                    except Exception:
                        continue
                if ep_df is None:
                    import io
                    raw = open(ep_path,'rb').read()
                    for enc in ['utf-8-sig','utf-8','gbk','latin1']:
                        try:
                            text = raw.decode(enc)
                            ep_df = pd.read_csv(io.StringIO(text))
                            break
                        except Exception:
                            continue
                if ep_df is not None and 'Average_Power_kW' in ep_df.columns:
                    grp = ep_df.groupby('Run')
                    hppo_extra = pd.DataFrame({
                        'Average Power per Run': grp['Average_Power_kW'].mean(),
                        'Average Station Utilization per Run': grp['Station_Utilization_%'].mean() if 'Station_Utilization_%' in ep_df.columns else np.nan,
                        'Violation Rate per Run': grp['Violation_Rate_%'].mean() if 'Violation_Rate_%' in ep_df.columns else np.nan
                    }).reset_index()
            # merge
            if hppo_extra is not None:
                merged = hppo_df.merge(hppo_extra, left_on='Run', right_on='Run', how='left')
            else:
                merged = hppo_df
            per_run_data['hppo'] = merged
    # Placeholders (future) for lirl / pdqn if per-run available
    # per_run_data['lirl'] = ...
    # per_run_data['pdqn'] = ...

    # 计算 p 值（仅对有数据的算法对）
    algorithms_with_per_run = list(per_run_data.keys())
    from itertools import combinations
    pval_rows = []
    for m in per_run_metrics:
        # 收集该指标的有效算法数据
        series_map = {}
        for alg in algorithms_with_per_run:
            df_alg = per_run_data[alg]
            if m in df_alg.columns:
                series = df_alg[m].dropna()
                if len(series) > 1:
                    series_map[alg] = series
        for a,b in combinations(series_map.keys(),2):
            a_vals = series_map[a]
            b_vals = series_map[b]
            # t-test (Welch)
            try:
                t_p = stats.ttest_ind(a_vals, b_vals, equal_var=False).pvalue
            except Exception:
                t_p = np.nan
            # Mann-Whitney (需要至少1个非零差异)
            try:
                mw_p = stats.mannwhitneyu(a_vals, b_vals, alternative='two-sided').pvalue
            except Exception:
                mw_p = np.nan
            pval_rows.append({'Metric':m,'AlgA':a,'AlgB':b,'t_test_p':t_p,'mw_p':mw_p})
    pval_df = pd.DataFrame(pval_rows)
    pval_csv = os.path.join(BASE_DIR,'compare_pairwise_pvalues.csv')
    if not pval_df.empty:
        pval_df.to_csv(pval_csv,index=False,encoding='utf-8-sig')
        print(f"已保存显著性检验结果: {pval_csv}")
    else:
        print("[INFO] 未生成显著性检验结果（缺少 per-run 数据）")

    # 绘制柱状图并添加简易显著性星号（仅对有 per-run 的算法间比较；当前仅 hppo 一个 -> 无法比较）
    bar_handles = []
    for i, alg in enumerate(algs):
        bar_container = ax4.bar(x + i*width, bar_df.loc[metrics_order, alg].values, width=width, label=alg.upper())
        bar_handles.append(bar_container)
        for xi, val in zip(x + i*width, bar_df.loc[metrics_order, alg].values):
            ax4.text(xi, val, f"{val:.1f}", ha='center', va='bottom', fontsize=7, rotation=90)
    # 如果未来有多算法 per-run，可在此添加显著性标注逻辑（基于 pval_df）
    ax4.set_xticks(x + width*(len(algs)-1)/2)
    ax4.set_xticklabels(metrics_order, fontproperties='SimHei', rotation=25, ha='right')
    ax4.set_ylabel('原始均值', fontproperties='SimHei')
    ax4.set_title('算法原始指标均值分组柱状图', fontproperties='SimHei')
    ax4.legend()
    plt.tight_layout()
    out_bars = os.path.join(BASE_DIR, 'compare_bars.png')
    plt.savefig(out_bars, dpi=200)
    print(f"已保存分组柱状图: {out_bars}")

    # ====== 额外：多子图雷达（整合三种视图） ======
    fig_multi, axes_multi = plt.subplots(1,3,subplot_kw={'polar':True}, figsize=(18,6))
    # 视图1：主归一 (norm)
    view_data = [
        ('归一化(反向违规)', norm, 'norm'),
        ('原始占比(value/max)', scaled, 'raw'),
        ('Soft-Floor', soft_norm, 'soft')
    ]
    angle_cache = {}
    for axm,(title_mat, mat_df, tag) in zip(axes_multi, view_data):
        labs = mat_df.index.tolist()
        angs = angle_cache.get(len(labs))
        if angs is None:
            angs = np.linspace(0,2*np.pi,len(labs),endpoint=False).tolist(); angs += angs[:1]
            angle_cache[len(labs)] = angs
        for alg in compare_df.columns:
            if alg not in mat_df.columns: continue
            vals = mat_df[alg].tolist(); vals += vals[:1]
            axm.plot(angs, vals, linewidth=2, label=alg.upper())
            axm.fill(angs, vals, alpha=0.12)
        axm.set_xticks(angs[:-1])
        axm.set_xticklabels(labs, fontproperties='SimHei', fontsize=9)
        axm.set_yticklabels([])
        axm.set_title(title_mat, fontproperties='SimHei')
    axes_multi[-1].legend(loc='upper right', bbox_to_anchor=(1.35,1.05))
    plt.tight_layout()
    multi_out = os.path.join(BASE_DIR,'compare_radar_subplots.png')
    plt.savefig(multi_out,dpi=220)
    print(f"已保存多子图雷达图: {multi_out}")

    # 另外输出 scores 统计
    score_stats_rows = []
    for alg, sub in ALG_DIRS.items():
        folder = os.path.join(BASE_DIR, sub)
        stats = load_scores_stats(alg, folder)
        if stats:
            stats['Algorithm'] = alg
            score_stats_rows.append(stats)
    if score_stats_rows:
        score_df = pd.DataFrame(score_stats_rows)
        out_scores_csv = os.path.join(BASE_DIR, 'compare_scores_summary.csv')
        score_df.to_csv(out_scores_csv, index=False, encoding='utf-8-sig')
        print(f"已保存 scores 统计: {out_scores_csv}")
        print(score_df)

if __name__ == '__main__':
    main()
