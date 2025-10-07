#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simplified radar plot script.

Function: produce a single performance radar chart for algorithms lirl / hppo / pdqn.
Scaling: per metric value / max(metric) across algorithms (no inversion for violation rate).

Metrics (English):
    Success Rate per Run
    Average Revenue per Run
    Average Energy per Run
    Average Power per Run
    Average Station Utilization per Run
    Violation Rate per Run

Output:
    compare_radar_simple.png

Usage:
    python compare_algorithms_simple.py [--show-labels]
Option:
    --show-labels   显示六个指标轴标签（默认隐藏），可用于导出带指标文字的版本。
"""
from __future__ import annotations
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import argparse

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ALG_DIRS = {'lirl':'lirl','hppo':'hppo','pdqn':'pdqn'}
TARGET_METRICS = [
    'Success Rate per Run',
    'Average Revenue per Run',
    'Average Energy per Run',
    'Average Power per Run',
    'Average Station Utilization per Run',
    'Violation Rate per Run'
]

# ----------------- Helpers -----------------

def load_overall_summary_generic(folder: str) -> pd.DataFrame | None:
    csvs = glob.glob(os.path.join(folder,'*overall_summary*.csv'))
    csvs = [p for p in csvs if 'vehicle_flow' not in os.path.basename(p)]
    if not csvs:
        return None
    path = sorted(csvs)[-1]
    # 尝试多编码
    for enc in ['utf-8-sig','utf-8','gbk','latin1']:
        try:
            df = pd.read_csv(path, encoding=enc)
            if 'Metric' in df.columns:
                return df
        except Exception:
            continue
    # 二进制兜底
    import io
    raw = open(path,'rb').read()
    for enc in ['utf-8-sig','utf-8','gbk','latin1']:
        try:
            text = raw.decode(enc)
            df = pd.read_csv(io.StringIO(text))
            if 'Metric' in df.columns:
                return df
        except Exception:
            continue
    return None

def extract_metric_values(df: pd.DataFrame, metrics: list[str]) -> dict:
    vals = {}
    for m in metrics:
        row = df[df['Metric']==m]
        if not row.empty:
            try:
                vals[m] = float(row['Mean'].values[0])
            except Exception:
                vals[m] = np.nan
        else:
            vals[m] = np.nan
    return vals

# ----------------- Main -----------------

def parse_args():
    p = argparse.ArgumentParser(description='Simple radar plot for algorithms.')
    p.add_argument('--show-labels', action='store_true', help='显示指标标签')
    return p.parse_args()

def main():
    args = parse_args()
    show_metric_labels = args.show_labels
    data_matrix = {}
    for alg, sub in ALG_DIRS.items():
        folder = os.path.join(BASE_DIR, sub)
        df = load_overall_summary_generic(folder)
        if df is None:
            print(f"[WARN] 未找到 {alg} overall summary，跳过")
            continue
        data_matrix[alg] = extract_metric_values(df, TARGET_METRICS)
    if not data_matrix:
        print("[ERROR] 无任何算法数据，终止")
        return

    # 构建 DataFrame: index=Metric, columns=Algorithm
    rows = []
    for m in TARGET_METRICS:
        row = {'Metric': m}
        for alg in data_matrix.keys():
            row[alg] = data_matrix[alg].get(m, np.nan)
        rows.append(row)
    df_metrics = pd.DataFrame(rows).set_index('Metric')

    # 缺失填补(用该指标现有值均值)
    for m in df_metrics.index:
        if df_metrics.loc[m].isna().any():
            mean_val = df_metrics.loc[m].mean(skipna=True)
            df_metrics.loc[m] = df_metrics.loc[m].fillna(mean_val)

    # 归一化: value / max(每指标)
    scaled = df_metrics.copy()
    for m in scaled.index:
        mx = scaled.loc[m].max()
        if mx > 0:
            scaled.loc[m] = scaled.loc[m] / mx
        else:
            scaled.loc[m] = 0.0

    # 雷达图
    metrics_order = scaled.index.tolist()
    N = len(metrics_order)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    # High-contrast, colorblind-friendly palette (Okabe-Ito subset, slightly deepened)
    colors = {
        'lirl': '#005B9A',      # deep blue (darker than #0072B2 for stronger line contrast)
        'hppo': '#D55E00',      # vermilion
        'pdqn': '#009E73'       # bluish green
    }

    # Helper to lighten color for fills (keep outlines vivid)
    def lighten(hex_color: str, factor: float = 0.35) -> str:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return '#' + hex_color
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r_l = int(r + (255 - r) * factor)
        g_l = int(g + (255 - g) * factor)
        b_l = int(b + (255 - b) * factor)
        return f"#{r_l:02X}{g_l:02X}{b_l:02X}"

    # Distinct line styles & markers for grayscale / print differentiation
    style_map = {
        'lirl': {'ls': '-',  'marker': 'o'},
        'hppo': {'ls': '--', 'marker': 's'},
        'pdqn': {'ls': ':',  'marker': '^'},
    }

    # Prepare raw values for annotation
    raw_values = df_metrics.loc[metrics_order]

    # Plot each algorithm and add numeric annotations (retain legend only)
    alg_list = list(scaled.columns)
    placed_texts = []  # store (angle, radius, Text object)
    for idx, alg in enumerate(alg_list):
        vals_scaled = scaled.loc[metrics_order, alg].tolist(); vals_scaled += vals_scaled[:1]
        base_color = colors.get(alg, 'gray')
        line_style = style_map.get(alg, {}).get('ls', '-')
        marker = style_map.get(alg, {}).get('marker', 'o')
        ax.plot(angles, vals_scaled,
                label=alg.upper(),
                linewidth=2.6,
                linestyle=line_style,
                marker=marker,
                markersize=7,
                markerfacecolor=base_color,
                markeredgecolor='white',
                markeredgewidth=1.1,
                color=base_color)
        ax.fill(angles, vals_scaled, alpha=0.12, color=lighten(base_color, 0.55))
        # numeric annotations INSIDE each vertex (inward radial offset)
        for angle_single, metric, sval in zip(angles[:-1], metrics_order, vals_scaled[:-1]):
            raw_val = raw_values.at[metric, alg]
            if isinstance(raw_val, (int,float,np.floating)) and np.isfinite(raw_val):
                if abs(raw_val) < 1e-12:
                    txt = "0.00"
                elif abs(raw_val) >= 1e3:
                    txt = f"{raw_val:,.0f}"
                elif abs(raw_val) >= 1:
                    txt = f"{raw_val:.2f}"
                elif abs(raw_val) >= 1e-2:
                    txt = f"{raw_val:.3f}"
                else:
                    txt = f"{raw_val:.2e}".replace('e+00','e0').replace('e-00','e0')
            else:
                txt = 'NA'
            # inward offset: bring text inside polygon (deeper for later algorithms)
            inward = 0.06 + idx*0.05
            r_offset = max(0.02, sval - inward)
            # small angular jitter still to reduce stacking
            jitter_dir = -1 if (idx + metrics_order.index(metric)) % 2 == 0 else 1
            jitter = jitter_dir * (0.01 + idx*0.003)
            ang_j = angle_single + jitter
            # Horizontal alignment: decide left/right based on angle (convert to degrees)
            deg = (np.degrees(ang_j) + 360) % 360
            if 90 < deg < 270:
                ha_align = 'right'
                tangential_shift = -0.008
            else:
                ha_align = 'left'
                tangential_shift = 0.008
            # Convert small tangential shift: adjust angle slightly for positioning while keeping label within interior
            ang_j_shifted = ang_j + tangential_shift
            try:
                import matplotlib.patheffects as pe
                peff=[pe.withStroke(linewidth=1.3, foreground='white')]
            except Exception:
                peff=[]
            # initial text placement
            t_obj = ax.text(ang_j_shifted, r_offset, txt, ha=ha_align, va='center', fontsize=14,
                            fontname='Arial', fontweight='bold', color=colors.get(alg,'gray'), path_effects=peff)
            placed_texts.append((ang_j, r_offset, t_obj))

    # ---- Simple collision avoidance (radial push) ----
    fig.canvas.draw()  # needed to get renderer for bbox sizes
    changed = True
    iteration = 0
    while changed and iteration < 6:
        changed = False
        iteration += 1
        for i in range(len(placed_texts)):
            for j in range(i+1, len(placed_texts)):
                ang_i, r_i, ti = placed_texts[i]
                ang_j, r_j, tj = placed_texts[j]
                # Only check if angles close (to avoid heavy trig) or radial distance small
                if abs(ang_i - ang_j) > 0.25:  # about 14 degrees threshold
                    continue
                bb_i = ti.get_window_extent(renderer=fig.canvas.get_renderer())
                bb_j = tj.get_window_extent(renderer=fig.canvas.get_renderer())
                if bb_i.overlaps(bb_j):
                    # push the inner (smaller radius) one further inward if possible
                    if r_i > r_j:
                        # push j inward
                        new_r = max(0.01, r_j - 0.03)
                        tj.set_position((ang_j, new_r))
                        placed_texts[j] = (ang_j, new_r, tj)
                    else:
                        new_r = max(0.01, r_i - 0.03)
                        ti.set_position((ang_i, new_r))
                        placed_texts[i] = (ang_i, new_r, ti)
                    changed = True

    # Extend radial limit to allow annotations outside 1.0
    ax.set_ylim(0, 1.0)
    # ---------------- Nature-style minimalist grid ----------------
    # Remove polar frame
    ax.spines['polar'].set_visible(False)
    # Set custom radial ticks (not labeled) for subtle structure
    ring_levels = [0.25, 0.50, 0.75, 1.0]
    for r in ring_levels:
        ax.plot(np.linspace(0, 2*np.pi, 400), [r]*400, color='#B8B8B8', linewidth=0.6, alpha=0.9, zorder=1)
    # Optional faint center cross (disabled to keep clean)
    # Style angular gridlines uniformly
    ax.set_thetagrids([])
    # Reduce default grid visibility
    ax.grid(False)
    # Draw custom angular spokes
    for ang in angles[:-1]:
        # thicker spoke for better visibility
        ax.plot([ang, ang], [0, 1.0], color='#9E9E9E', linewidth=0.9, alpha=0.65, zorder=1)
        # subtle endpoint background dot (under markers)
        ax.scatter([ang], [1.0], s=26, color='white', edgecolors='none', zorder=1)

    ax.set_yticklabels([])
    # Metric labels (optional via toggle)
    if show_metric_labels:
        # custom placement slightly outside radius 1
        label_radius = 1.05
        for ang, metric in zip(angles[:-1], metrics_order):
            deg = (np.degrees(ang) + 360) % 360
            if 90 < deg < 270:
                ha_align = 'right'
                tangential_shift = -0.01
            else:
                ha_align = 'left'
                tangential_shift = 0.01
            ang_shifted = ang + tangential_shift
            try:
                import matplotlib.patheffects as pe
                peff=[pe.withStroke(linewidth=1.6, foreground='white')]
            except Exception:
                peff=[]
            ax.text(ang_shifted, label_radius, metric, ha=ha_align, va='center',
                    fontsize=12, fontname='Arial', fontweight='bold', color='#333333', path_effects=peff)
        # hide default ticks
        ax.set_xticks([])
    else:
        # keep metrics hidden
        ax.set_xticks([])
    # Removed title per user request
    # Reposition legend to bottom-right close to plot
    ax.legend(loc='lower right', bbox_to_anchor=(1.05,-0.05), prop={'family':'Arial','size':11,'weight':'bold'})
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR,'compare_radar_simple.png')
    plt.savefig(out_path, dpi=200)
    print(f"已保存: {out_path}")

if __name__ == '__main__':
    main()
