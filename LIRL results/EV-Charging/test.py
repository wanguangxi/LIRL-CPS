# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 指标顺序（可调整）
metrics = [
    "Success Rate per Run",
    "Average Revenue per Run",
    "Average Energy per Run",
    "Average Power per Run",
    "Average Station Utilization per Run",
    "Violation Rate per Run"
]

# 三个算法的均值（与你给出的数据一致）
data = {
    "lirl": {
        "Success Rate per Run": 90.40,
        "Average Revenue per Run": 12455.53,
        "Average Energy per Run": 11368.43,
        "Average Power per Run": 146.95,
        "Average Station Utilization per Run": 52.11,
        "Violation Rate per Run": 0.00
    },
    "pdqn": {
        "Success Rate per Run": 69.68,
        "Average Revenue per Run": 9597.37,
        "Average Energy per Run": 8778.05,
        "Average Power per Run": 126.28,
        "Average Station Utilization per Run": 51.75,
        "Violation Rate per Run": 24.77
    },
    "hppo": {
        "Success Rate per Run": 46.515933,
        "Average Revenue per Run": 6157.976000,
        "Average Energy per Run": 5681.516000,
        "Average Power per Run": 109.434722,
        "Average Station Utilization per Run": 44.564000,
        "Violation Rate per Run": 57.339596
    }
}

algorithms = list(data.keys())

def build_matrix(metrics, data):
    """返回 shape=(len(algorithms), len(metrics)) 的矩阵"""
    mat = []
    for alg in algorithms:
        mat.append([data[alg][m] for m in metrics])
    return np.array(mat)

values_raw = build_matrix(metrics, data)  # 行=算法, 列=指标

def normalize_per_metric(arr):
    """对每个指标列做 min-max 归一化，列相等则置 0.5"""
    arr_norm = arr.copy().astype(float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mn, mx = col.min(), col.max()
        if mx > mn:
            arr_norm[:, j] = (col - mn) / (mx - mn)
        else:
            arr_norm[:, j] = 0.5
    return arr_norm

# 归一化
values_norm = normalize_per_metric(values_raw)

# （版本1）对“Violation Rate per Run”做反向（越低越好）
violation_index = metrics.index("Violation Rate per Run")
values_norm_inverted = values_norm.copy()
values_norm_inverted[:, violation_index] = 1 - values_norm_inverted[:, violation_index]

# 绘制雷达图函数
def plot_radar(values, metrics, title, filename, note_violation=False):
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    colors = {
        "lirl": "#1f77b4",
        "pdqn": "#2ca02c",
        "hppo": "#ff7f0e"
    }

    for i, alg in enumerate(algorithms):
        vals = values[i].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=alg.upper(), linewidth=2, marker='o')
        ax.fill(angles, vals, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontproperties='SimHei', fontsize=10)
    ax.set_yticklabels([])
    ax.set_title(title, fontproperties='SimHei')
    if note_violation:
        ax.text(0.0, -1.2, "注：Violation Rate 已反向处理（越低越好）", 
                transform=ax.transAxes, ha='left', va='center', fontsize=9, fontproperties='SimHei')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.08))
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"已生成: {filename}")
    plt.close(fig)

# 绘制（版本1：违规率反向）
plot_radar(values_norm_inverted, metrics,
           "算法性能雷达图 (归一化 + 违规率反向)", 
           "radar_normalized_inverted_violation.png",
           note_violation=True)

# 绘制（版本2：纯归一化，不反向，展示原始方向）
plot_radar(values_norm, metrics,
           "算法性能雷达图 (归一化原始方向)", 
           "radar_normalized_raw_direction.png",
           note_violation=False)

# 也打印一下归一化后的矩阵，便于核对

print(values_norm)