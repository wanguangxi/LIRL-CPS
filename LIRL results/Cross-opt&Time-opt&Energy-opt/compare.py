#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 experiments/result 目录下的三种优化模式（以子文件夹名为模式名）进行对比分析：
1) 四种规模（scale_a/b/c/d）下、九种配置下的得分（scores.npy 后 100 组）的对比；
2) 四种规模下、九种配置下的总能耗（energy_data.json）对比；
3) 四种规模下、九种配置下的完工时间（gantt_data.json 的最大 end_time）对比。

输出：
- CSV 汇总：experiments/compare_reports/summary_metrics.csv
- 每个规模各三张分组柱状图：scores、energy、makespan（PNG）

运行：
- 直接执行本文件：python experiments/compare.py
- 可通过命令行参数指定输入/输出目录。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 尝试使用 matplotlib 生成图表；若不可用则仅输出 CSV
_PLOTTING_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端，便于服务器/脚本环境保存图片
    import matplotlib.pyplot as plt
    # 尝试导入 seaborn for swarm plots
    try:
        import seaborn as sns
        SEABORN_AVAILABLE = True
    except ImportError:
        SEABORN_AVAILABLE = False
    # 设定更美观的风格与Arial字体
    try:
        plt.style.use('seaborn-v0_8')
    except Exception:
        try:
            plt.style.use('seaborn')
        except Exception:
            pass
    # Set Arial as the primary font
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    _PLOTTING_AVAILABLE = False
    SEABORN_AVAILABLE = False


# Nature/Okabe-Ito 颜色方案（颜色盲友好，Nature 推荐）
NATURE_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Scale to Job-Resource notation mapping
SCALE_MAPPING = {
    "scale_a": "J10R3",
    "scale_b": "J20R3",
    "scale_c": "J50R5",
    "scale_d": "J100R5",
}


# ----------------------------- 数据结构 -----------------------------
@dataclass
class Record:
    mode: str          # 优化模式（目录名）
    scale: str         # 规模（scale_a/b/c/d）
    config: str        # 配置（如 scale_0.1+0.9 等）
    score_mean: float  # scores.npy 后 100 组的均值（若不足 100，取全部可用）
    score_std: float   # 对应标准差
    score_tail: Optional[List[float]]  # 用于箱线图的末尾样本
    total_energy: float  # energy_data.json 中 total_energy 的求和（若无，尝试 work+idle）
    makespan: float      # gantt_data.json 中所有条目的最大 end_time


# ----------------------------- 工具函数 -----------------------------
def _read_scores_mean_std_tail(scores_path: Path, tail_n: int = 100) -> Tuple[float, float, List[float], int]:
    """读取scores并返回均值、标准差、末尾样本，以及最高分的索引"""
    arr = np.load(scores_path)
    if arr.ndim > 1:
        # 若是二维（如 [episodes, metrics]），尽量取第一列；若形状不明，拉平
        try:
            arr = arr[:, 0]
        except Exception:
            arr = arr.ravel()
    if arr.size == 0:
        return float("nan"), float("nan"), [], -1
    n = min(tail_n, arr.size)
    tail = arr[-n:]
    # 找到末尾n个中的最高分索引（相对于整个数组）
    best_idx_in_tail = np.argmax(tail)
    best_idx_overall = len(arr) - n + best_idx_in_tail
    return float(np.mean(tail)), float(np.std(tail)), tail.astype(float).tolist(), best_idx_overall


def _read_energy_total(energy_json: Path) -> float:
    """读取能耗总和（用作fallback）"""
    with energy_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 优先 total_energy 的总和；否则 work+idle 的总和
    if "total_energy" in data and isinstance(data["total_energy"], list):
        return float(np.sum(np.array(data["total_energy"], dtype=float)))
    work = np.sum(np.array(data.get("work_energy", []), dtype=float))
    idle = np.sum(np.array(data.get("idle_energy", []), dtype=float))
    if work + idle > 0:
        return float(work + idle)
    # 兜底：将所有数值字段相加
    total = 0.0
    for v in data.values():
        if isinstance(v, (int, float)):
            total += float(v)
        elif isinstance(v, list):
            try:
                total += float(np.sum(np.array(v, dtype=float)))
            except Exception:
                pass
    return float(total)


def _read_makespan(gantt_json: Path) -> float:
    """读取makespan（用作fallback）"""
    with gantt_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # data 预计为一个包含任务条目的列表，每个有 end_time
    max_end = 0.0
    if isinstance(data, list):
        for item in data:
            try:
                end_t = float(item.get("end_time", 0.0))
            except Exception:
                end_t = 0.0
            if end_t > max_end:
                max_end = end_t
    return float(max_end)


def _read_energy_total_at_index(energy_json: Path, idx: int) -> float:
    """读取指定索引位置的能耗值"""
    with energy_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 尝试获取特定索引的能耗
    if "total_energy" in data and isinstance(data["total_energy"], list):
        if 0 <= idx < len(data["total_energy"]):
            return float(data["total_energy"][idx])
    
    # 如果没有total_energy，尝试work+idle
    work_energy = 0.0
    idle_energy = 0.0
    if "work_energy" in data and isinstance(data["work_energy"], list):
        if 0 <= idx < len(data["work_energy"]):
            work_energy = float(data["work_energy"][idx])
    if "idle_energy" in data and isinstance(data["idle_energy"], list):
        if 0 <= idx < len(data["idle_energy"]):
            idle_energy = float(data["idle_energy"][idx])
    
    if work_energy + idle_energy > 0:
        return work_energy + idle_energy
    
    # 如果索引无效，返回总和作为fallback
    return _read_energy_total(energy_json)


def _read_makespan_at_index(gantt_json: Path, idx: int) -> float:
    """读取指定索引位置的makespan值"""
    with gantt_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 如果data是列表的列表（每个episode一个列表）
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], list):
            # data是[episode][tasks]格式
            if 0 <= idx < len(data):
                episode_data = data[idx]
                max_end = 0.0
                for item in episode_data:
                    try:
                        end_t = float(item.get("end_time", 0.0))
                    except Exception:
                        end_t = 0.0
                    if end_t > max_end:
                        max_end = end_t
                return max_end
        elif isinstance(data[0], dict):
            # data是单个episode的任务列表（第一个元素是dict说明是任务）
            # 尝试查看是否有makespans字段
            return _read_makespan(gantt_json)
    
    # 尝试查看是否有makespans字段（某些格式可能直接存储）
    if isinstance(data, dict):
        if "makespans" in data and isinstance(data["makespans"], list):
            if 0 <= idx < len(data["makespans"]):
                return float(data["makespans"][idx])
    
    # Fallback to original method
    return _read_makespan(gantt_json)


def _collect_modes(base_dir: Path) -> List[str]:
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir()])


def _collect_scales(mode_dir: Path) -> List[str]:
    # 只保留类似 scale_ 前缀的目录，并按名称排序
    return sorted([p.name for p in mode_dir.iterdir() if p.is_dir() and p.name.startswith("scale_")])


def _collect_configs(scale_dir: Path) -> List[str]:
    # 子目录即为不同配置（九组）
    return sorted([p.name for p in scale_dir.iterdir() if p.is_dir()])


# ----------------------------- 主流程 -----------------------------
def gather_records(base_dir: Path, tail_n: int = 100) -> List[Record]:
    records: List[Record] = []
    modes = _collect_modes(base_dir)
    for mode in modes:
        mode_dir = base_dir / mode
        scales = _collect_scales(mode_dir)
        for scale in scales:
            scale_dir = mode_dir / scale
            configs = _collect_configs(scale_dir)
            for cfg in configs:
                cfg_dir = scale_dir / cfg
                scores_path = cfg_dir / "scores.npy"
                energy_json = cfg_dir / "energy_data.json"
                gantt_json = cfg_dir / "gantt_data.json"

                if not scores_path.exists() or not energy_json.exists() or not gantt_json.exists():
                    # 跳过不完整数据
                    continue

                try:
                    score_mean, score_std, score_tail, best_idx = _read_scores_mean_std_tail(scores_path, tail_n=tail_n)
                except Exception as e:
                    print(f"Error reading scores from {scores_path}: {e}")
                    score_mean, score_std, score_tail, best_idx = float("nan"), float("nan"), [], -1

                # 使用最高分对应的能耗和makespan
                try:
                    if best_idx >= 0:
                        total_energy = _read_energy_total_at_index(energy_json, best_idx)
                    else:
                        total_energy = _read_energy_total(energy_json)
                except Exception as e:
                    print(f"Error reading energy from {energy_json}: {e}")
                    total_energy = float("nan")

                try:
                    if best_idx >= 0:
                        makespan = _read_makespan_at_index(gantt_json, best_idx)
                    else:
                        makespan = _read_makespan(gantt_json)
                except Exception as e:
                    print(f"Error reading makespan from {gantt_json}: {e}")
                    makespan = float("nan")
                
                # Debug print to check values
                if not np.isnan(total_energy) or not np.isnan(makespan):
                    print(f"Found data for {mode}/{scale}/{cfg}: energy={total_energy:.2f}, makespan={makespan:.2f}, best_idx={best_idx}")

                records.append(
                    Record(
                        mode=mode,
                        scale=scale,
                        config=cfg,
                        score_mean=score_mean,
                        score_std=score_std,
                        score_tail=score_tail,
                        total_energy=total_energy,
                        makespan=makespan,
                    )
                )
    return records


def ensure_output_dir(dir_path: Path) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_csv(records: List[Record], out_csv: Path) -> None:
    import csv
    ensure_output_dir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", "scale", "config",
            "score_mean_last100", "score_std_last100",
            "total_energy", "makespan",
        ])
        for r in records:
            writer.writerow([
                r.mode, r.scale, r.config,
                f"{r.score_mean:.6f}", f"{r.score_std:.6f}",
                f"{r.total_energy:.6f}", f"{r.makespan:.6f}",
            ])


def get_max_values(records: List[Record]) -> Tuple[float, float]:
    """
    获取所有记录中的最大makespan和最大energy值
    
    Args:
        records: 数据记录列表
        
    Returns:
        (max_makespan, max_energy): 最大的makespan和energy值
    """
    max_makespan = float('-inf')
    max_energy = float('-inf')
    
    for record in records:
        # 检查makespan
        if not np.isnan(record.makespan):
            max_makespan = max(max_makespan, record.makespan)
        
        # 检查energy
        if not np.isnan(record.total_energy):
            max_energy = max(max_energy, record.total_energy)
    
    # 如果没有找到有效值，返回NaN
    if max_makespan == float('-inf'):
        max_makespan = float('nan')
    if max_energy == float('-inf'):
        max_energy = float('nan')
    
    return max_makespan, max_energy


def get_max_values_by_scale(records: List[Record]) -> Dict[str, Tuple[float, float]]:
    """
    按scale分组，获取每个scale中的最大makespan和最大energy值
    
    Args:
        records: 数据记录列表
        
    Returns:
        Dict[scale, (max_makespan, max_energy)]: 每个scale的最大值
    """
    by_scale: Dict[str, List[Record]] = {}
    for r in records:
        by_scale.setdefault(r.scale, []).append(r)
    
    result = {}
    for scale, scale_records in by_scale.items():
        max_makespan, max_energy = get_max_values(scale_records)
        result[scale] = (max_makespan, max_energy)
        
    return result


def get_max_values_by_mode(records: List[Record]) -> Dict[str, Tuple[float, float]]:
    """
    按mode分组，获取每个mode中的最大makespan和最大energy值
    
    Args:
        records: 数据记录列表
        
    Returns:
        Dict[mode, (max_makespan, max_energy)]: 每个mode的最大值
    """
    by_mode: Dict[str, List[Record]] = {}
    for r in records:
        by_mode.setdefault(r.mode, []).append(r)
    
    result = {}
    for mode, mode_records in by_mode.items():
        max_makespan, max_energy = get_max_values(mode_records)
        result[mode] = (max_makespan, max_energy)
        
    return result


def print_max_values_summary(records: List[Record]) -> None:
    """
    打印最大值的汇总信息
    
    Args:
        records: 数据记录列表
    """
    # 全局最大值
    global_max_makespan, global_max_energy = get_max_values(records)
    print("\n" + "="*60)
    print("Maximum Values Summary")
    print("="*60)
    print(f"\nGlobal Maximum Values:")
    print(f"  Max Makespan: {global_max_makespan:.2f}")
    print(f"  Max Energy:   {global_max_energy:.2f}")
    
    # 按scale的最大值
    scale_max = get_max_values_by_scale(records)
    if scale_max:
        print(f"\nMaximum Values by Scale:")
        for scale in sorted(scale_max.keys()):
            max_makespan, max_energy = scale_max[scale]
            job_resource = SCALE_MAPPING.get(scale, scale)
            print(f"  {job_resource:8s} - Makespan: {max_makespan:8.2f}, Energy: {max_energy:8.2f}")
    
    # 按mode的最大值
    mode_max = get_max_values_by_mode(records)
    if mode_max:
        print(f"\nMaximum Values by Mode:")
        for mode in sorted(mode_max.keys()):
            max_makespan, max_energy = mode_max[mode]
            print(f"  {mode.capitalize():12s} - Makespan: {max_makespan:8.2f}, Energy: {max_energy:8.2f}")
    
    print("="*60 + "\n")


def _plot_scores_boxplot(
    out_png: Path,
    title: str,
    configs: List[str],
    modes: List[str],
    score_tails: Dict[str, List[List[float]]],
) -> None:
    if not _PLOTTING_AVAILABLE:
        return

    ensure_output_dir(out_png.parent)

    # Ensure Arial font is used
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    x = np.arange(len(configs), dtype=float)
    n = max(1, len(modes))
    group_width = 0.8
    gap = group_width / n

    # Nature journal specifications: smaller figure with higher DPI
    fig, ax = plt.subplots(figsize=(8, 5))  # Reduced size for Nature format

    # Nature color palette
    colors = NATURE_PALETTE

    handles = []
    labels = []

    for j, mode in enumerate(modes):
        mode_data = []  # 按配置排列的数据
        positions = []
        for i, cfg in enumerate(configs):
            tails = score_tails.get(mode, [])
            arr = tails[i] if i < len(tails) else []
            # 确保为一维数组；若为空，用包含 NaN 的数组占位，避免 boxplot 失败
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.size == 0:
                arr = np.array([np.nan])
            mode_data.append(arr)
            positions.append(x[i] + (j - (n - 1) / 2.0) * gap)

        bp = ax.boxplot(
            mode_data,
            positions=positions,
            widths=gap * 0.8,  # Slightly narrower boxes
            patch_artist=True,
            showmeans=False,  # Remove mean markers for cleaner look
            notch=False,  # No notches for cleaner appearance
            manage_ticks=False,
            # Nature-style box properties
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            medianprops=dict(linewidth=1.5, color='black'),
            flierprops=dict(marker='o', markersize=3, markeredgewidth=0.5),
        )
        
        color = colors[j % len(colors)]
        # Set consistent colors for all box elements
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(1.0)
            patch.set_edgecolor(color)  # Use same color for edge
        for whisk in bp['whiskers']:
            whisk.set_color(color)  # Use same color for whiskers
            whisk.set_linewidth(1.0)
        for cap in bp['caps']:
            cap.set_color(color)  # Use same color for caps
            cap.set_linewidth(1.0)
        for median in bp['medians']:
            median.set_color('black')  # Keep median black for contrast
            median.set_linewidth(1.5)
        for flier in bp.get('fliers', []):
            flier.set(marker='o', markersize=3, markerfacecolor=color, 
                     markeredgecolor=color, alpha=0.5, markeredgewidth=0.5)  # Use same color for outliers

        handles.append(bp['boxes'][0])
        labels.append(mode.capitalize())  # Capitalize first letter

    # Nature journal formatting with Arial font
    ax.set_title(title, fontsize=12, fontweight='normal', pad=10, fontfamily='Arial')
    ax.set_ylabel("Performance", fontsize=11, fontfamily='Arial')
    ax.set_xlabel("Weight Factor α, β=1-α", fontsize=11, fontfamily='Arial')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10, fontfamily='Arial')
    
    # Legend with Nature style and Arial font
    if handles:
        legend = ax.legend(handles, labels, frameon=False, loc='best', 
                          fontsize=10, ncol=min(3, len(labels)))
        for text in legend.get_texts():
            text.set_fontfamily('Arial')
    
    # Set Arial for tick labels
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Grid with Nature style
    ax.grid(axis="y", linestyle="-", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')  # Higher DPI for Nature
    plt.close(fig)


def _plot_scores_swarm(
    out_png: Path,
    title: str,
    configs: List[str],
    modes: List[str],
    score_tails: Dict[str, List[List[float]]],
) -> None:
    """绘制蜂群图展示得分分布，包含四分位数、中位数和包络线"""
    if not _PLOTTING_AVAILABLE:
        return
    
    if not SEABORN_AVAILABLE:
        print("Seaborn not available, falling back to boxplot")
        _plot_scores_boxplot(out_png, title, configs, modes, score_tails)
        return

    ensure_output_dir(out_png.parent)

    # Ensure Arial font is used
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # Prepare data for seaborn
    data_list = []
    for mode in modes:
        mode_tails = score_tails.get(mode, [])
        for i, config in enumerate(configs):
            if i < len(mode_tails):
                scores = mode_tails[i]
                if len(scores) > 0:
                    for score in scores:
                        if np.isfinite(score):
                            data_list.append({
                                'Config': config,
                                'Mode': mode.capitalize(),  # Capitalize for display
                                'Score': score,
                                'ConfigIndex': i  # Add index for envelope plotting
                            })
    
    if not data_list:
        print("No valid data for swarm plot")
        return
    
    import pandas as pd
    df = pd.DataFrame(data_list)
    
    # Nature journal specifications
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get capitalized mode names for display
    display_modes = [mode.capitalize() for mode in modes]
    
    # Nature color palette
    colors = NATURE_PALETTE[:len(display_modes)]
    
    # Create swarm plot
    sns.swarmplot(
        data=df,
        x='Config',
        y='Score',
        hue='Mode',
        palette=dict(zip(display_modes, colors)),
        size=3,  # Point size
        alpha=0.6,  # Lower opacity to see overlapping elements
        ax=ax,
        dodge=True,
        hue_order=display_modes
    )
    
    # Calculate and plot envelope curves for each mode
    for mode_idx, (original_mode, display_mode) in enumerate(zip(modes, display_modes)):
        mode_df = df[df['Mode'] == display_mode]
        color = colors[mode_idx]
        
        # Calculate statistics for each config
        envelope_upper = []
        envelope_lower = []
        envelope_median = []
        envelope_q1 = []
        envelope_q3 = []
        x_positions = []
        median_x_positions = []  # Separate positions for median markers
        
        for i, config in enumerate(configs):
            config_data = mode_df[mode_df['Config'] == config]['Score'].values
            if len(config_data) > 0:
                envelope_upper.append(np.max(config_data))
                envelope_lower.append(np.min(config_data))
                envelope_median.append(np.median(config_data))
                envelope_q1.append(np.percentile(config_data, 25))
                envelope_q3.append(np.percentile(config_data, 75))
                
                # Calculate x position for this mode and config (for envelopes)
                n_modes = len(modes)
                dodge_width = 0.8 / n_modes
                x_pos = i + (mode_idx - (n_modes - 1) / 2) * dodge_width
                x_positions.append(x_pos)
                
                # For median markers, use the exact x-axis position (no dodge)
                median_x_positions.append(i)
        
        if x_positions:
            # Plot envelope (min-max range)
            ax.fill_between(x_positions, envelope_lower, envelope_upper, 
                           alpha=0.15, color=color, label=f'{display_mode} range')
            
            # Plot interquartile range (Q1-Q3)
            ax.fill_between(x_positions, envelope_q1, envelope_q3, 
                           alpha=0.25, color=color)
            
            # Plot median line connecting the dodged positions
            ax.plot(x_positions, envelope_median, 
                   color=color, linewidth=1.5, linestyle='-', alpha=0.5)
            
            # Plot median markers at vertical x-axis positions
            ax.scatter(median_x_positions, envelope_median,
                      color=color, s=50, marker='D', 
                      edgecolors='black', linewidths=1.5,
                      alpha=0.9, zorder=10,
                      label=f'{display_mode} median')
            
            # Plot Q1 and Q3 lines
            ax.plot(x_positions, envelope_q1, 
                   color=color, linewidth=1, linestyle='--', alpha=0.7)
            ax.plot(x_positions, envelope_q3, 
                   color=color, linewidth=1, linestyle='--', alpha=0.7)
    
    # Add horizontal lines for reference (optional)
    # This helps to compare performance across different configs
    ax.axhline(y=df['Score'].median(), color='gray', linestyle=':', 
              linewidth=0.8, alpha=0.5, label='Overall median')
    
    # Nature journal formatting with Arial font
    ax.set_title(title, fontsize=12, fontweight='normal', pad=10, fontfamily='Arial')
    ax.set_ylabel("Performance", fontsize=11, fontfamily='Arial')
    ax.set_xlabel("Weight Factor α, β=1-α", fontsize=11, fontfamily='Arial')
    
    # Custom legend to avoid duplication
    handles, labels = ax.get_legend_handles_labels()
    # Filter to keep only mode names and median lines
    filtered_handles = []
    filtered_labels = []
    seen_modes = set()
    
    for h, l in zip(handles, labels):
        if ' range' not in l and 'Overall median' not in l:
            mode_name = l.replace(' median', '')
            if mode_name not in seen_modes:
                filtered_handles.append(h)
                filtered_labels.append(mode_name)
                seen_modes.add(mode_name)
    
    ax.legend(filtered_handles, filtered_labels, 
             frameon=False, loc='best', fontsize=10, 
             ncol=min(3, len(display_modes)), title=None)
    
    # Add text annotations for quartiles
    ax.text(0.02, 0.98, 'Diamond markers: Median\nShaded areas: IQR (Q1-Q3)\nLight shade: Min-Max range', 
           transform=ax.transAxes, fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set Arial for tick labels
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Grid with Nature style
    ax.grid(axis="y", linestyle="-", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_grouped_bars(
    out_png: Path,
    title: str,
    x_labels: List[str],
    series_dict: Dict[str, List[float]],
    ylabel: str,
    yerr_dict: Optional[Dict[str, List[float]]] = None,
    rotate_xtick: Optional[int] = None,
    annotate: bool = False,
) -> None:
    if not _PLOTTING_AVAILABLE:
        return

    ensure_output_dir(out_png.parent)
    
    # Ensure Arial font is used
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    modes = list(series_dict.keys())
    x = np.arange(len(x_labels))
    n = len(modes)
    width = 0.8 / max(n, 1)

    # Nature journal specifications
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, mode in enumerate(modes):
        vals = np.array(series_dict[mode], dtype=float)
        offs = (i - (n - 1) / 2) * width
        yerr = None
        if yerr_dict and mode in yerr_dict:
            yerr = np.array(yerr_dict[mode], dtype=float)
        # 使用 Nature 调色板分配颜色
        color = NATURE_PALETTE[i % len(NATURE_PALETTE)]
        bars = ax.bar(
            x + offs, vals, width * 0.9,  # Slightly narrower bars
            label=mode.capitalize(),  # Capitalize first letter
            yerr=yerr, capsize=3,
            edgecolor='black', linewidth=0.8,
            color=color,
            alpha=0.8,
        )
        if annotate:
            try:
                ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=8, rotation=90)
            except Exception:
                pass

    ax.set_title(title, fontsize=12, fontweight='normal', pad=10, fontfamily='Arial')
    ax.set_ylabel(ylabel, fontsize=11, fontfamily='Arial')
    ax.set_xlabel("Weight Factor α, β=1-α", fontsize=11, fontfamily='Arial')
    ax.set_xticks(x)
    if rotate_xtick is not None:
        ax.set_xticklabels(x_labels, rotation=rotate_xtick, ha="right", fontsize=10, fontfamily='Arial')
    else:
        ax.set_xticklabels(x_labels, fontsize=10, fontfamily='Arial')
    
    # Add a note about using best-score values
    ax.text(0.02, 0.98, 'Values from best-scoring episode', 
           transform=ax.transAxes, fontsize=8, verticalalignment='top', style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend with Nature style and Arial font
    legend = ax.legend(frameon=False, loc='best', fontsize=10, ncol=min(3, len(modes)))
    for text in legend.get_texts():
        text.set_fontfamily('Arial')
    
    # Set Arial for tick labels
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Grid with Nature style
    ax.grid(axis="y", linestyle="-", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4)

    # 为正值数据留出顶部空间
    all_vals = np.array([v for arr in series_dict.values() for v in arr], dtype=float)
    if np.isfinite(all_vals).any():
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size and np.nanmin(finite) >= 0:
            ymax = float(np.nanmax(finite)) * 1.1 if finite.size else None
            if ymax and ymax > 0:
                ax.set_ylim(0, ymax)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')  # Higher DPI for Nature
    plt.close(fig)


def _plot_grouped_bars_normalized(
    out_png: Path,
    title: str,
    x_labels: List[str],
    series_dict: Dict[str, List[float]],
    ylabel: str,
    baseline_mode: Optional[str] = None,
    yerr_dict: Optional[Dict[str, List[float]]] = None,
    rotate_xtick: Optional[int] = None,
    annotate: bool = False,
) -> None:
    """绘制归一化的分组柱状图，使用统一基准进行归一化"""
    if not _PLOTTING_AVAILABLE:
        return

    ensure_output_dir(out_png.parent)
    
    # Ensure Arial font is used
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    modes = list(series_dict.keys())
    
    # 选择基准模式（如果未指定，使用第一个模式）
    if baseline_mode is None or baseline_mode not in modes:
        baseline_mode = modes[0]
    
    # 计算基准值（基准模式的平均值）
    baseline_values = np.array(series_dict[baseline_mode], dtype=float)
    valid_baseline = baseline_values[~np.isnan(baseline_values)]
    if len(valid_baseline) > 0:
        baseline = np.mean(valid_baseline)
    else:
        baseline = 1.0  # 避免除零
    
    # 归一化所有数据
    normalized_dict = {}
    for mode in modes:
        vals = np.array(series_dict[mode], dtype=float)
        normalized_dict[mode] = vals / baseline if baseline > 0 else vals
    
    x = np.arange(len(x_labels))
    n = len(modes)
    width = 0.8 / max(n, 1)

    # Nature journal specifications
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, mode in enumerate(modes):
        vals = normalized_dict[mode]
        offs = (i - (n - 1) / 2) * width
        yerr = None
        if yerr_dict and mode in yerr_dict:
            yerr_vals = np.array(yerr_dict[mode], dtype=float)
            yerr = yerr_vals / baseline if baseline > 0 else yerr_vals
        
        # 使用 Nature 调色板分配颜色
        color = NATURE_PALETTE[i % len(NATURE_PALETTE)]
        bars = ax.bar(
            x + offs, vals, width * 0.9,  # Slightly narrower bars
            label=mode.capitalize(),  # Capitalize first letter
            yerr=yerr, capsize=3,
            edgecolor='black', linewidth=0.8,
            color=color,
            alpha=0.8,
        )
        if annotate:
            try:
                ax.bar_label(bars, fmt='%.2f', padding=2, fontsize=8, rotation=90)
            except Exception:
                pass

    # Add a horizontal line at y=1 to show the baseline
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(len(x_labels) - 0.5, 1.02, f'Baseline: {baseline_mode.capitalize()}', 
           ha='right', va='bottom', fontsize=8, color='gray', style='italic')

    ax.set_title(title, fontsize=12, fontweight='normal', pad=10, fontfamily='Arial')
    ax.set_ylabel(f"Normalized {ylabel}", fontsize=11, fontfamily='Arial')
    ax.set_xlabel("Weight Factor α, β=1-α", fontsize=11, fontfamily='Arial')
    ax.set_xticks(x)
    if rotate_xtick is not None:
        ax.set_xticklabels(x_labels, rotation=rotate_xtick, ha="right", fontsize=10, fontfamily='Arial')
    else:
        ax.set_xticklabels(x_labels, fontsize=10, fontfamily='Arial')
    
    # Add notes about normalization
    ax.text(0.02, 0.98, f'Normalized to {baseline_mode.capitalize()} average\nValues from best-scoring episode', 
           transform=ax.transAxes, fontsize=8, verticalalignment='top', style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend with Nature style and Arial font
    legend = ax.legend(frameon=False, loc='best', fontsize=10, ncol=min(3, len(modes)))
    for text in legend.get_texts():
        text.set_fontfamily('Arial')
    
    # Set Arial for tick labels
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Grid with Nature style
    ax.grid(axis="y", linestyle="-", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4)
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')  # Higher DPI for Nature
    plt.close(fig)


def _plot_weighted_combination(
    out_png: Path,
    title: str,
    x_labels: List[str],
    alpha_values: List[float],
    energies: Dict[str, List[float]],
    makespans: Dict[str, List[float]],
) -> None:
    """绘制加权组合图：makespan*α + energy*(1-α)"""
    if not _PLOTTING_AVAILABLE:
        return

    ensure_output_dir(out_png.parent)
    
    # Ensure Arial font is used
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    modes = list(energies.keys())
    
    # Nature journal specifications
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect all scores for y-axis scaling
    all_weighted_scores = []
    
    # Calculate weighted combinations for each mode
    for i, mode in enumerate(modes):
        mode_energies = np.array(energies[mode], dtype=float)
        mode_makespans = np.array(makespans[mode], dtype=float)
        
        # Skip if all values are NaN
        if np.all(np.isnan(mode_energies)) or np.all(np.isnan(mode_makespans)):
            continue
        
        # Get max values for normalization
        current_max_makespan = np.nanmax(mode_makespans)
        current_max_energy = np.nanmax(mode_energies)
        print(f"Mode: {mode}, Max Makespan: {current_max_makespan:.2f}, Max Energy: {current_max_energy:.2f}")
        
        # Calculate weighted combination with normalization
        weighted_scores = []
        for j, alpha in enumerate(alpha_values):
            if not np.isnan(mode_makespans[j]) and not np.isnan(mode_energies[j]):
                # Normalized weighted combination
                normalized_makespan = mode_makespans[j] / current_max_makespan if current_max_makespan > 0 else 0
                normalized_energy = mode_energies[j] / current_max_energy if current_max_energy > 0 else 0
                score = normalized_makespan * alpha + normalized_energy * (1 - alpha)
                weighted_scores.append(score)
                all_weighted_scores.append(score)
            else:
                weighted_scores.append(np.nan)
        
        # Plot line with markers
        color = NATURE_PALETTE[i % len(NATURE_PALETTE)]
        ax.plot(alpha_values, weighted_scores, 
               marker='o', markersize=6, 
               linewidth=2, 
               color=color, 
               label=mode.capitalize(),
               alpha=0.8)
        
        # Add markers for emphasis
        ax.scatter(alpha_values, weighted_scores, 
                  s=30, color=color, 
                  edgecolors='black', linewidths=0.5,
                  zorder=5, alpha=0.9)
    
    # Add vertical reference line at α=0.5 (equal weight)
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Nature journal formatting with Arial font
    ax.set_title(title, fontsize=12, fontweight='normal', pad=10, fontfamily='Arial')
    ax.set_ylabel("Normalized Weighted Score\n(Makespan/max(M)×α + Energy/max(E)×(1-α))", 
                 fontsize=11, fontfamily='Arial')
    ax.set_xlabel("Weight Factor α", fontsize=11, fontfamily='Arial')
    
    # Set x-axis
    ax.set_xticks(alpha_values)
    ax.set_xticklabels([f"{a:.1f}" for a in alpha_values], fontsize=10, fontfamily='Arial')
    
    # Legend with Nature style and Arial font
    legend = ax.legend(frameon=False, loc='best', fontsize=10, ncol=min(3, len(modes)))
    for text in legend.get_texts():
        text.set_fontfamily('Arial')
    
    # Set Arial for tick labels
    for label in ax.get_xticklabels():
        label.set_fontfamily('Arial')
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Grid with Nature style
    ax.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10, width=0.8, length=4)
    
    # Auto-scale y-axis based on actual normalized scores
    if all_weighted_scores:
        valid_scores = [s for s in all_weighted_scores if not np.isnan(s)]
        if valid_scores:
            y_min = min(valid_scores)
            y_max = max(valid_scores)
            y_range = y_max - y_min
            
            # Add some padding
            y_padding = 0.1 * y_range if y_range > 0 else 0.1
            ax.set_ylim(max(0, y_min - y_padding), min(1, y_max + y_padding))
            
            # Add equal weight reference text at appropriate position
            ax.text(0.5, y_min - 0.5 * y_padding if y_min - 0.5 * y_padding > 0 else 0.02, 
                   'Equal Weight', ha='center', va='bottom', fontsize=9, color='gray', style='italic')
    else:
        # Default range if no valid scores
        ax.set_ylim(0, 1)
    
    # Add annotation about the formula at bottom-left corner
    ax.text(0.02, 0.02, 'Normalized Score = (Makespan/max(M))×α + (Energy/max(E))×(1-α)\nUsing best-scoring episodes, normalized by mode', 
           transform=ax.transAxes, fontsize=8, verticalalignment='bottom', style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_per_scale(records: List[Record], out_dir: Path, plot_type: str = "box") -> List[Path]:
    """
    为每个 scale 生成三张图（scores/energy/makespan）。返回生成的图片路径列表。
    
    Args:
        records: 数据记录列表
        out_dir: 输出目录
        plot_type: 得分图类型，"box" for boxplot, "swarm" for swarm plot
    """
    if not _PLOTTING_AVAILABLE:
        return []

    images: List[Path] = []
    # 按 scale 分组
    by_scale: Dict[str, List[Record]] = {}
    for r in records:
        by_scale.setdefault(r.scale, []).append(r)

    for scale, recs in sorted(by_scale.items(), key=lambda kv: kv[0]):
        # Get the job-resource notation for this scale
        job_resource = SCALE_MAPPING.get(scale, scale)
        
        # 收集该 scale 下的所有配置名称（并确保在不同模式下覆盖一致），并按 alpha 排序
        raw_configs = sorted({r.config for r in recs})
        alpha_pairs: List[Tuple[str, float]] = []
        ok = True
        for c in raw_configs:
            a = _parse_alpha(c)
            if a is None:
                ok = False
                break
            alpha_pairs.append((c, a))
        if ok and alpha_pairs:
            alpha_pairs.sort(key=lambda t: t[1])
            configs = [c for c, _ in alpha_pairs]
            alpha_labels = [f"α={a:.1f}" for _, a in alpha_pairs]
            alpha_values = [a for _, a in alpha_pairs]
        else:
            configs = raw_configs
            alpha_labels = [_pretty_config_label(c) for c in configs]
            alpha_values = list(np.linspace(0.1, 0.9, len(configs)))  # Default alpha values
        modes = sorted({r.mode for r in recs})

        # 准备数据：每个模式一条序列，按照 configs 顺序填充值
        score_means: Dict[str, List[float]] = {m: [] for m in modes}
        score_stds: Dict[str, List[float]] = {m: [] for m in modes}
        score_tails: Dict[str, List[List[float]]] = {m: [] for m in modes}
        energies: Dict[str, List[float]] = {m: [] for m in modes}
        makespans: Dict[str, List[float]] = {m: [] for m in modes}

        # 建立查找表
        idx: Dict[Tuple[str, str], Record] = {(r.mode, r.config): r for r in recs}
        for cfg in configs:
            for m in modes:
                r = idx.get((m, cfg))
                if r is None:
                    score_means[m].append(np.nan)
                    score_stds[m].append(0.0)
                    score_tails[m].append([])
                    energies[m].append(np.nan)
                    makespans[m].append(np.nan)
                else:
                    score_means[m].append(r.score_mean)
                    score_stds[m].append(r.score_std)
                    score_tails[m].append(r.score_tail or [])
                    energies[m].append(r.total_energy)
                    makespans[m].append(r.makespan)

        # 输出三张图 - 使用新的标题格式
        # 1) Performance Distribution - 根据plot_type选择箱线图或蜂群图
        if plot_type == "swarm":
            img_scores = out_dir / f"{scale}_scores_swarm.png"
            _plot_scores_swarm(
                out_png=img_scores,
                title=f"{job_resource} - Performance Distribution",
                configs=alpha_labels,
                modes=modes,
                score_tails=score_tails,
            )
        else:
            img_scores = out_dir / f"{scale}_scores.png"
            _plot_scores_boxplot(
                out_png=img_scores,
                title=f"{job_resource} - Performance Distribution",
                configs=alpha_labels,
                modes=modes,
                score_tails=score_tails,
            )
        images.append(img_scores)

        # 原始能耗图
        img_energy = out_dir / f"{scale}_energy.png"
        _plot_grouped_bars(
            img_energy,
            title=f"{job_resource} - Energy Consumption (Best-Score Episodes)",
            x_labels=alpha_labels,
            series_dict=energies,
            yerr_dict=None,
            ylabel="Energy Consumption",
            rotate_xtick=None,
            annotate=False,
        )
        images.append(img_energy)
        
        # 归一化能耗图
        img_energy_norm = out_dir / f"{scale}_energy_normalized.png"
        _plot_grouped_bars_normalized(
            img_energy_norm,
            title=f"{job_resource} - Normalized Energy Consumption",
            x_labels=alpha_labels,
            series_dict=energies,
            ylabel="Energy Consumption",
            baseline_mode=modes[0] if modes else None,  # 使用第一个模式作为基准
            rotate_xtick=None,
            annotate=False,
        )
        images.append(img_energy_norm)

        # 原始makespan图
        img_makespan = out_dir / f"{scale}_makespan.png"
        _plot_grouped_bars(
            img_makespan,
            title=f"{job_resource} - Makespan (Best-Score Episodes)",
            x_labels=alpha_labels,
            series_dict=makespans,
            yerr_dict=None,
            ylabel="Makespan",
            rotate_xtick=None,
            annotate=False,
        )
        images.append(img_makespan)
        
        # 归一化makespan图
        img_makespan_norm = out_dir / f"{scale}_makespan_normalized.png"
        _plot_grouped_bars_normalized(
            img_makespan_norm,
            title=f"{job_resource} - Normalized Makespan",
            x_labels=alpha_labels,
            series_dict=makespans,
            ylabel="Makespan",
            baseline_mode=modes[0] if modes else None,  # 使用第一个模式作为基准
            rotate_xtick=None,
            annotate=False,
        )
        images.append(img_makespan_norm)
        
        # Add weighted combination plot
        img_weighted = out_dir / f"{scale}_weighted_combination.png"
        _plot_weighted_combination(
            img_weighted,
            title=f"{job_resource} - Weighted Combination (Makespan×α + Energy×(1-α))",
            x_labels=alpha_labels,
            alpha_values=alpha_values,
            energies=energies,
            makespans=makespans,
        )
        images.append(img_weighted)

    return images


def _pretty_config_label(cfg: str) -> str:
    """更紧凑易读的配置标签：去掉前缀并在加号处分行。"""
    if cfg.startswith('scale_'):
        cfg = cfg[len('scale_'):]
    return cfg.replace('+', '\n+\n')


def _parse_alpha(cfg: str) -> Optional[float]:
    """从配置名解析 alpha（格式示例：scale_0.1+0.9）。失败返回 None。"""
    s = cfg
    if s.startswith('scale_'):
        s = s[len('scale_'):]
    # 取加号前的第一段
    part = s.split('+', 1)[0]
    try:
        return float(part)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare modes across scales/configs using scores/energy/gantt data.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).parent / "result"),
        help="Input directory (contains subdirectories for different modes), default: experiments/result",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "compare_reports"),
        help="Output directory (CSV and charts), default: experiments/compare_reports",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=100,
        help="Number of last episodes to use from scores.npy, default: 100",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["box", "swarm", "both"],
        default="swarm",
        help="Type of plot for score distribution: box (boxplot), swarm (swarm plot), or both",
    )
    args = parser.parse_args()

    base_dir = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    ensure_output_dir(out_dir)

    if not base_dir.exists():
        raise SystemExit(f"Input directory does not exist: {base_dir}")

    # 收集记录
    records = gather_records(base_dir, tail_n=args.tail)
    if not records:
        raise SystemExit("未在输入目录中发现有效的数据文件（scores.npy/energy_data.json/gantt_data.json）。")

    # 打印最大值汇总信息
    print_max_values_summary(records)
    
    # 保存 CSV 汇总
    out_csv = out_dir / "summary_metrics.csv"
    save_csv(records, out_csv)
    
    # 生成图表
    images: List[Path] = []
    if _PLOTTING_AVAILABLE:
        if args.plot_type == "both":
            # Generate both boxplot and swarm plot
            images.extend(plot_per_scale(records, out_dir, plot_type="box"))
            images.extend(plot_per_scale(records, out_dir, plot_type="swarm"))
        else:
            images = plot_per_scale(records, out_dir, plot_type=args.plot_type)
    else:
        print("matplotlib 不可用，将仅生成 CSV 文件。")

    print("分析完成：")
    print(f"- CSV: {out_csv}")
    if images:
        for p in images:
            print(f"- 图表: {p}")
    
    if SEABORN_AVAILABLE and args.plot_type in ["swarm", "both"]:
        print("\n蜂群图已生成。蜂群图能更直观地展示每个数据点的分布情况。")
    elif args.plot_type in ["swarm", "both"] and not SEABORN_AVAILABLE:
        print("\n注意：Seaborn未安装，无法生成蜂群图。请运行: pip install seaborn pandas")









def _pretty_config_label(cfg: str) -> str:
    """更紧凑易读的配置标签：去掉前缀并在加号处分行。"""
    if cfg.startswith('scale_'):
        cfg = cfg[len('scale_'):]
    return cfg.replace('+', '\n+\n')


def _parse_alpha(cfg: str) -> Optional[float]:
    """从配置名解析 alpha（格式示例：scale_0.1+0.9）。失败返回 None。"""
    s = cfg
    if s.startswith('scale_'):
        s = s[len('scale_'):]
    # 取加号前的第一段
    part = s.split('+', 1)[0]
    try:
        return float(part)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare modes across scales/configs using scores/energy/gantt data.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).parent / "result"),
        help="Input directory (contains subdirectories for different modes), default: experiments/result",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "compare_reports"),
        help="Output directory (CSV and charts), default: experiments/compare_reports",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=100,
        help="Number of last episodes to use from scores.npy, default: 100",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["box", "swarm", "both"],
        default="swarm",
        help="Type of plot for score distribution: box (boxplot), swarm (swarm plot), or both",
    )
    args = parser.parse_args()

    base_dir = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    ensure_output_dir(out_dir)

    if not base_dir.exists():
        raise SystemExit(f"Input directory does not exist: {base_dir}")

    # 收集记录
    records = gather_records(base_dir, tail_n=args.tail)
    if not records:
        raise SystemExit("未在输入目录中发现有效的数据文件（scores.npy/energy_data.json/gantt_data.json）。")

    # 打印最大值汇总信息
    print_max_values_summary(records)
    
    # 保存 CSV 汇总
    out_csv = out_dir / "summary_metrics.csv"
    save_csv(records, out_csv)
    
    # 生成图表
    images: List[Path] = []
    if _PLOTTING_AVAILABLE:
        if args.plot_type == "both":
            # Generate both boxplot and swarm plot
            images.extend(plot_per_scale(records, out_dir, plot_type="box"))
            images.extend(plot_per_scale(records, out_dir, plot_type="swarm"))
        else:
            images = plot_per_scale(records, out_dir, plot_type=args.plot_type)
    else:
        print("matplotlib 不可用，将仅生成 CSV 文件。")

    print("分析完成：")
    print(f"- CSV: {out_csv}")
    if images:
        for p in images:
            print(f"- 图表: {p}")
    
    if SEABORN_AVAILABLE and args.plot_type in ["swarm", "both"]:
        print("\n蜂群图已生成。蜂群图能更直观地展示每个数据点的分布情况。")
    elif args.plot_type in ["swarm", "both"] and not SEABORN_AVAILABLE:
        print("\n注意：Seaborn未安装，无法生成蜂群图。请运行: pip install seaborn pandas")







if __name__ == "__main__":
    main()

