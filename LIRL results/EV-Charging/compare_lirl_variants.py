#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Radar comparison: original LIRL vs LIRL_fenceng.

Reference: compare_algorithms_simple.py (same normalization per metric: value / max among compared algorithms).

Usage:
    python compare_lirl_variants.py [--show-labels]

Output:
    compare_lirl_variants.png
"""
from __future__ import annotations
import os, glob, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ALG_DIRS = {'lirl':'lirl','lirl_fenceng':'LIRL_fenceng'}  # mapping new variant
TARGET_METRICS = [
    'Success Rate per Run',
    'Average Revenue per Run',
    'Average Energy per Run',
    'Average Power per Run',
    'Average Station Utilization per Run',
    'Violation Rate per Run'
]

def parse_args():
    p = argparse.ArgumentParser(description='Radar: LIRL vs LIRL_fenceng')
    p.add_argument('--show-labels', action='store_true', help='显示指标标签')
    p.add_argument('--bg-theme', choices=['none','cool','warm'], default='none', help='背景渐变主题 (A: 已默认关闭)')
    p.add_argument('--bg-intensity', type=float, default=1.0, help='背景强度(用于渐变，当前默认无背景)')
    return p.parse_args()

def load_overall_summary(folder: str):
    csvs = glob.glob(os.path.join(folder,'*overall_summary*.csv'))
    csvs = [p for p in csvs if 'vehicle_flow' not in os.path.basename(p)]
    if not csvs:
        return None
    path = sorted(csvs)[-1]
    for enc in ['utf-8-sig','utf-8','gbk','latin1']:
        try:
            df = pd.read_csv(path, encoding=enc)
            if 'Metric' in df.columns:
                return df
        except Exception:
            continue
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

def extract_metric_values(df: pd.DataFrame, metrics):
    out = {}
    for m in metrics:
        row = df[df['Metric']==m]
        if not row.empty:
            try:
                out[m] = float(row['Mean'].values[0])
            except Exception:
                out[m] = np.nan
        else:
            out[m] = np.nan
    return out

def main():
    args = parse_args()
    show_metric_labels = args.show_labels

    data_matrix = {}
    for alg, sub in ALG_DIRS.items():
        folder = os.path.join(BASE_DIR, sub)
        df = load_overall_summary(folder)
        if df is None:
            print(f'[WARN] 没有找到 {alg} overall summary, 跳过')
            continue
        data_matrix[alg] = extract_metric_values(df, TARGET_METRICS)
    if not data_matrix:
        print('[ERROR] 没有任何数据，终止')
        return

    rows = []
    for m in TARGET_METRICS:
        row = {'Metric': m}
        for alg in data_matrix.keys():
            row[alg] = data_matrix[alg].get(m, np.nan)
        rows.append(row)
    df_metrics = pd.DataFrame(rows).set_index('Metric')

    for m in df_metrics.index:
        if df_metrics.loc[m].isna().any():
            mean_val = df_metrics.loc[m].mean(skipna=True)
            df_metrics.loc[m] = df_metrics.loc[m].fillna(mean_val)

    scaled = df_metrics.copy()
    for m in scaled.index:
        mx = scaled.loc[m].max()
        scaled.loc[m] = scaled.loc[m] / mx if mx > 0 else 0.0

    metrics_order = scaled.index.tolist()
    N = len(metrics_order)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]

    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    # ---- Configurable radial gradient background ----
    if args.bg_theme != 'none':  # Option A: user requested removal, default now none
        try:
            import numpy as _np
            grad_res = 600
            gx = _np.linspace(-1,1,grad_res)
            gy = _np.linspace(-1,1,grad_res)
            XX, YY = _np.meshgrid(gx, gy)
            RR = _np.sqrt(XX**2 + YY**2)
            RR_norm = _np.clip(RR,0,1)
            if args.bg_theme == 'cool':
                # center soft blue (slightly desaturated) -> edge light gray
                # Interpolate between two RGB colors
                c_center = _np.array([0.78, 0.86, 0.96])  # slightly deeper
                c_edge   = _np.array([0.94, 0.95, 0.97])
            else:  # warm
                c_center = _np.array([0.985, 0.965, 0.915])  # richer ivory
                c_edge   = _np.array([0.93, 0.93, 0.935])
            # Intensity scaling
            intensity = max(0.0, min(2.0, args.bg_intensity))
            # Blend curve: sharpen center contrast when intensity>1
            curve = RR_norm**(0.9 + 0.4*(2-intensity))
            RGB = c_center*(1-curve)[:, :, None] + c_edge*(curve)[:, :, None]
            # Alpha stronger if intensity high
            alpha = (1 - (0.85 - 0.25*(intensity-1))*RR_norm)
            # outside circle transparent
            mask = RR>1
            alpha[mask] = 0
            # stack RGBA
            BG = _np.dstack([RGB, alpha])
            ax.imshow(BG, origin='lower', extent=(-1,1,-1,1), zorder=0, interpolation='bilinear')
        except Exception:
            pass

    # Match palette style from compare_algorithms_simple.py (deepened Okabe-Ito subset)
    colors = {
        'lirl': '#005B9A',      # same deep blue used there
        'lirl_fenceng': '#D55E00'  # vermilion (reuse HPPO hue to contrast LIRL)
    }

    # We will compute fill colors by lightening the outline color (like simple script)
    # Option B: stronger yet elegant fill visibility
    FILL_ALPHA = 0.22

    def lighten(hex_color: str, factor: float = 0.35) -> str:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return '#' + hex_color
        r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
        r_l = int(r + (255 - r) * factor); g_l = int(g + (255 - g) * factor); b_l = int(b + (255 - b) * factor)
        return f"#{r_l:02X}{g_l:02X}{b_l:02X}"

    style_map = {
        'lirl': {'ls': '-', 'marker': 'o'},
        'lirl_fenceng': {'ls': '--', 'marker': 's'}
    }

    raw_values = df_metrics.loc[metrics_order]
    placed_texts = []
    import matplotlib.patheffects as pe
    legend_name_map = {
        'lirl': 'Cross-opt (LIRL)',
        'lirl_fenceng': 'Hierarchical-opt'
    }
    for idx, alg in enumerate(list(scaled.columns)):
        vals_scaled = scaled.loc[metrics_order, alg].tolist(); vals_scaled += vals_scaled[:1]
        base_color = colors.get(alg, 'gray')
        line_style = style_map.get(alg, {}).get('ls', '-')
        marker = style_map.get(alg, {}).get('marker', 'o')
        fill_color = lighten(base_color, 0.65)  # lighter interior (Option B)
        display_label = legend_name_map.get(alg, alg.upper())
        ax.plot(angles, vals_scaled, label=display_label, linewidth=2.6, linestyle=line_style,  # match simple script (Option C)
                marker=marker, markersize=7, markerfacecolor=base_color, markeredgecolor='white',
                markeredgewidth=1.1, color=base_color)
        ax.fill(angles, vals_scaled, color=fill_color, alpha=FILL_ALPHA, zorder=2)
        for ang_single, metric, sval in zip(angles[:-1], metrics_order, vals_scaled[:-1]):
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
                    # very small but non-zero, keep scientific without showing +00 if near zero
                    txt = f"{raw_val:.2e}".replace('e+00','e0').replace('e-00','e0')
            else:
                txt = 'NA'
            inward = 0.06 + idx*0.05
            r_offset = max(0.02, sval - inward)
            jitter_dir = -1 if (idx + metrics_order.index(metric)) % 2 == 0 else 1
            jitter = jitter_dir * (0.01 + idx*0.003)
            ang_j = ang_single + jitter
            deg = (np.degrees(ang_j) + 360) % 360
            if 90 < deg < 270:
                ha_align = 'right'; tangential_shift = -0.008
            else:
                ha_align = 'left'; tangential_shift = 0.008
            ang_shifted = ang_j + tangential_shift
            t_obj = ax.text(ang_shifted, r_offset, txt, ha=ha_align, va='center', fontsize=14,
                            fontname='Arial', fontweight='bold', color=base_color,
                            path_effects=[pe.withStroke(linewidth=1.3, foreground='white')])
            placed_texts.append((ang_j, r_offset, t_obj))

    fig.canvas.draw()
    changed = True; iteration = 0
    while changed and iteration < 6:
        changed = False; iteration += 1
        for i in range(len(placed_texts)):
            for j in range(i+1, len(placed_texts)):
                ang_i, r_i, ti = placed_texts[i]; ang_j, r_j, tj = placed_texts[j]
                if abs(ang_i - ang_j) > 0.25: continue
                bb_i = ti.get_window_extent(renderer=fig.canvas.get_renderer())
                bb_j = tj.get_window_extent(renderer=fig.canvas.get_renderer())
                if bb_i.overlaps(bb_j):
                    if r_i > r_j:
                        new_r = max(0.01, r_j - 0.03); tj.set_position((ang_j, new_r)); placed_texts[j] = (ang_j, new_r, tj)
                    else:
                        new_r = max(0.01, r_i - 0.03); ti.set_position((ang_i, new_r)); placed_texts[i] = (ang_i, new_r, ti)
                    changed = True

    ax.set_ylim(0,1.0)
    # rings
    for r in [0.25,0.5,0.75,1.0]:
        ax.plot(np.linspace(0,2*np.pi,400), [r]*400, color='#B8B8B8', linewidth=0.6, alpha=0.9, zorder=1)
    ax.spines['polar'].set_visible(False)
    ax.set_thetagrids([]); ax.grid(False)
    for ang in angles[:-1]:
        ax.plot([ang, ang], [0,1.0], color='#9E9E9E', linewidth=0.9, alpha=0.65, zorder=1)
        ax.scatter([ang], [1.0], s=26, color='white', edgecolors='none', zorder=1)

    ax.set_yticklabels([])
    if show_metric_labels:
        label_radius = 1.05
        for ang, metric in zip(angles[:-1], metrics_order):
            deg = (np.degrees(ang) + 360) % 360
            if 90 < deg < 270:
                ha_align = 'right'; shift = -0.01
            else:
                ha_align = 'left'; shift = 0.01
            ang_shifted = ang + shift
            ax.text(ang_shifted, label_radius, metric, ha=ha_align, va='center', fontsize=12,
                    fontname='Arial', fontweight='bold', color='#333333',
                    path_effects=[pe.withStroke(linewidth=1.6, foreground='white')])
        ax.set_xticks([])
    else:
        ax.set_xticks([])

    ax.legend(loc='lower right', bbox_to_anchor=(1.05,-0.05), prop={'family':'Arial','size':11,'weight':'bold'})
    plt.tight_layout()
    out_path = os.path.join(BASE_DIR,'compare_lirl_variants.png')
    plt.savefig(out_path, dpi=220)
    print(f'已保存: {out_path}')

if __name__ == '__main__':
    main()
