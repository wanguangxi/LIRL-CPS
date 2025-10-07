"""
Result Analysis Tool - Directly read multi-run training results of algorithms from Compare folder
"""
import json
import os
import glob
import argparse
from pathlib import Path
import re

# 尝试导入可选依赖
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class CompareAnalyzer:
    def __init__(self, compare_dir: str = None):
        """
        Initialize result analyzer
        
        Args:
            compare_dir: Compare directory path, default is current directory
        """
        if compare_dir is None:
            compare_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.compare_dir = Path(compare_dir)
        print(f"Analysis directory: {self.compare_dir}")
        
        # Auto discover algorithm results
        self.algorithms = {}
        self.auto_discover_algorithms()
        
    def _normalize_scores(self, scores):
        """Normalize loaded scores into a 2D array [runs, episodes] by trimming to min length"""
        if not NUMPY_AVAILABLE:
            return scores
        try:
            arr = np.array(scores, dtype=object)
            if arr.dtype == object:
                runs = [np.asarray(r).astype(float).reshape(-1) for r in arr]
                if len(runs) == 0:
                    return np.zeros((0, 0))
                min_len = min(len(r) for r in runs) if all(len(r) > 0 for r in runs) else 0
                if min_len == 0:
                    return np.zeros((len(runs), 0))
                return np.stack([r[:min_len] for r in runs], axis=0)
            # If already numeric
            arr = np.asarray(scores)
            if arr.ndim == 1:
                return arr[np.newaxis, :]
            if arr.ndim >= 2:
                return arr[:, :arr.shape[1]]
            return arr
        except Exception as e:
            print(f"  - Score normalization failed: {e}")
            return np.array(scores, dtype=object)

    def auto_discover_algorithms(self):
        """Auto discover all algorithm results in Compare folder"""
        print("\nScanning algorithm results...")
        
        for item in self.compare_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if contains multi-run training results
                config_files = list(item.glob('config_*.json'))
                score_files = list(item.glob('*_all_scores_*.npy'))
                
                if not config_files:
                    print(f"Skip {item.name}: no config_*.json found")
                    continue
                if not score_files:
                    print(f"Skip {item.name}: no *_all_scores_*.npy found")
                    continue
                
                if NUMPY_AVAILABLE:
                    # Extract algorithm name
                    algo_name = self.extract_algorithm_name(item.name)
                    print(f"Found algorithm: {algo_name} -> {item.name}")
                    
                    try:
                        # Load data
                        config_file = config_files[0]
                        score_file = score_files[0]
                        
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        # Allow loading dtype=object arrays saved by numpy
                        scores_raw = np.load(score_file, allow_pickle=True)
                        scores = self._normalize_scores(scores_raw)
                        
                        self.algorithms[algo_name] = {
                            'path': item,
                            'config': config,
                            'scores': scores,
                            'config_file': config_file,
                            'score_file': score_file
                        }
                        
                        print(f"  - Config file: {config_file.name}")
                        print(f"  - Score file: {score_file.name}")
                        print(f"  - Number of runs: {len(scores)}")
                        print(f"  - Steps per run (trimmed to min length): {scores.shape[1] if scores.ndim == 2 else 0}")
                        
                    except Exception as e:
                        print(f"  - Loading failed ({item.name}): {e}")
                else:
                    print(f"Skip {item.name}: numpy unavailable")
        if not self.algorithms:
            print("No algorithm results found!")
            if not NUMPY_AVAILABLE:
                print("Note: numpy unavailable, cannot load .npy files")
        else:
            print(f"\nTotal found {len(self.algorithms)} algorithm results")
            
    def extract_algorithm_name(self, folder_name: str) -> str:
        """Extract algorithm name from folder name"""
        # Remove timestamp suffix
        name = re.sub(r'_multi_run_\d{8}_\d{6}$', '', folder_name)
        
        # Algorithm name mapping
        name_mapping = {
            'ddpg_lirl_pi': 'LIRL',
            'cpo': 'CPO+Mask',
            'hppo': 'HPPO+Mask',
            'hyar_vae': 'HyAR+Mask',
            "sac_lag": "SAC-Lag+Mask",
            "pdqn": "PDQN+Mask",

        }
        
        return name_mapping.get(name, name.upper())
        
    def analyze_convergence(self):
        """Analyze convergence performance"""
        if not NUMPY_AVAILABLE:
            print("numpy unavailable, skipping convergence analysis")
            return
            
        print("\n" + "="*60)
        print("Convergence Performance Analysis")
        print("="*60)
        
        for algo_name, data in self.algorithms.items():
            scores = data['scores']
            print(f"\n{algo_name}:")
            
            # Calculate final performance statistics
            final_scores = scores[:, -1]  # Last step scores
            mean_final = np.mean(final_scores)
            std_final = np.std(final_scores)
            
            print(f"  Final performance: {mean_final:.4f} ± {std_final:.4f}")
            print(f"  Best run: {np.max(final_scores):.4f}")
            print(f"  Worst run: {np.min(final_scores):.4f}")
            
            # Calculate convergence stability (std of last 10% steps)
            last_10_percent = int(len(scores[0]) * 0.1)
            if last_10_percent > 0:
                stability_scores = scores[:, -last_10_percent:]
                stability = np.mean(np.std(stability_scores, axis=1))
                print(f"  Convergence stability: {stability:.4f} (lower is more stable)")
                
        print("\n" + "-"*60)
                
    def compare_algorithms(self):
        """Compare algorithm performance"""
        if len(self.algorithms) < 2:
            print("Need at least 2 algorithms for comparison")
            return
            
        if not NUMPY_AVAILABLE:
            print("numpy unavailable, skipping algorithm comparison")
            return
            
        print("\n" + "="*60)
        print("Algorithm Performance Comparison")
        print("="*60)
        
        # Prepare data
        algo_names = list(self.algorithms.keys())
        final_performances = {}
        
        for algo_name, data in self.algorithms.items():
            scores = data['scores']
            final_performances[algo_name] = scores[:, -1]
            
        # Pairwise comparison
        for i in range(len(algo_names)):
            for j in range(i + 1, len(algo_names)):
                algo1, algo2 = algo_names[i], algo_names[j]
                perf1, perf2 = final_performances[algo1], final_performances[algo2]
                
                mean1, mean2 = np.mean(perf1), np.mean(perf2)
                std1, std2 = np.std(perf1), np.std(perf2)
                
                print(f"\n{algo1} vs {algo2}:")
                print(f"  {algo1}: {mean1:.4f} ± {std1:.4f}")
                print(f"  {algo2}: {mean2:.4f} ± {std2:.4f}")
                
                # Calculate difference
                diff = mean1 - mean2
                print(f"  Difference: {diff:.4f}")
                
                if abs(diff) > 0.01:  # Meaningful difference when > 0.01
                    if diff > 0:
                        print(f"  🏆 {algo1} performs better on average")
                    else:
                        print(f"  🏆 {algo2} performs better on average")
                else:
                    print("  📊 Average performance is comparable")
                    
                # Statistical significance test (if scipy available)
                if SCIPY_AVAILABLE:
                    try:
                        t_stat, p_value = stats.ttest_ind(perf1, perf2)
                        print(f"  Statistical test: t={t_stat:.4f}, p={p_value:.4f}")
                        
                        if p_value < 0.05:
                            print("  ✅ Statistically significant difference (p < 0.05)")
                        else:
                            print("  ❌ No statistical significance (p >= 0.05)")
                    except Exception as e:
                        print(f"  Statistical test failed: {e}")
                        
        print("\n" + "-"*60)
                        
    def plot_training_curves(self):
        """Plot training curves"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            print("matplotlib or numpy unavailable, skipping plotting")
            return

        print("\nPlotting training curves...")

        # Modern elegant color palette - inspired by Tailwind CSS and Material Design
        ELEGANT_PALETTE = [
            "#6366F1",  # Indigo - primary blue-purple
            "#F59E0B",  # Amber - warm orange
            "#10B981",  # Emerald - fresh green
            "#EC4899",  # Pink - vibrant pink
            "#14B8A6",  # Teal - cyan-green
            "#8B5CF6",  # Violet - purple
            "#F97316",  # Orange - pure orange
            "#06B6D4",  # Cyan - bright cyan
            "#84CC16",  # Lime - yellow-green
            "#A855F7",  # Purple - deep purple
        ]

        # Try to use a clean, elegant style
        try:
            import matplotlib as mpl
            from contextlib import contextmanager

            @contextmanager
            def _style_ctx():
                try:
                    with plt.style.context("seaborn-v0_8-whitegrid"):
                        yield
                except Exception:
                    yield

            with _style_ctx():
                plt.rcParams.update({
                    "figure.facecolor": "none",  # Transparent figure background
                    "axes.facecolor": "none",  # Transparent axes background
                    "savefig.facecolor": "none",  # Transparent when saving
                    "axes.grid": True,
                    "grid.color": "#E5E7EB",  # Softer grid lines
                    "grid.alpha": 0.8,
                    "grid.linewidth": 0.5,
                    "axes.edgecolor": "#D1D5DB",
                    "axes.linewidth": 1.2,
                    "axes.titleweight": "bold",
                    "axes.titlesize": 13,
                    "axes.labelsize": 11,
                    "axes.labelweight": "medium",
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.frameon": True,
                    "legend.facecolor": "white",
                    "legend.edgecolor": "#E5E7EB",
                    "legend.framealpha": 0.95,
                    "legend.borderpad": 0.8,
                    "font.family": "sans-serif",
                })

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), constrained_layout=True)
                
                # Set transparent background
                fig.patch.set_alpha(0.0)

                def _beautify_axis(ax):
                    # Enhanced axis styling
                    ax.minorticks_on()
                    ax.grid(which="minor", axis="both", alpha=0.15, linewidth=0.3, color="#F3F4F6")
                    ax.grid(which="major", axis="both", alpha=0.3, linewidth=0.5, color="#E5E7EB")
                    
                    # Modern spine styling
                    for spine in ["top", "right"]:
                        ax.spines[spine].set_visible(False)
                    for spine in ["left", "bottom"]:
                        ax.spines[spine].set_color("#9CA3AF")
                        ax.spines[spine].set_linewidth(0.8)
                    
                    # Set axes background to transparent
                    ax.patch.set_alpha(0.0)

                colors = ELEGANT_PALETTE
                color_idx = 0
                
                # Collect all final scores for determining x-axis range
                all_final_scores = []
                for algo_name, data in self.algorithms.items():
                    scores = data['scores']
                    if scores is not None and getattr(scores, "size", 0) > 0 and scores.ndim == 2 and scores.shape[1] > 0:
                        all_final_scores.extend(scores[:, -1])
                
                # Determine x-axis range for KDE
                if all_final_scores:
                    x_min = np.min(all_final_scores) - 0.02
                    x_max = np.max(all_final_scores) + 0.02
                    x_kde = np.linspace(x_min, x_max, 200)

                for algo_name, data in self.algorithms.items():
                    scores = data['scores']
                    # Skip if no data
                    if scores is None or getattr(scores, "size", 0) == 0:
                        continue
                    if scores.ndim != 2 or scores.shape[1] == 0:
                        continue

                    # Special elegant color for LIRL algorithm
                    if algo_name == 'LIRL':
                        color = "#EF4444"  # Modern red - more elegant than crimson
                        linewidth = 2.8  # Slightly thicker for emphasis
                        alpha = 1.0
                    else:
                        color = colors[color_idx % len(colors)]
                        color_idx += 1
                        linewidth = 2.2
                        alpha = 0.9

                    # Calculate mean and standard deviation
                    mean_scores = np.mean(scores, axis=0)
                    std_scores = np.std(scores, axis=0)
                    episodes = np.arange(len(mean_scores))

                    # Training curves with enhanced styling
                    line = ax1.plot(
                        episodes, mean_scores,
                        label=algo_name, color=color, linewidth=linewidth, alpha=alpha,
                        marker='', markersize=0,  # No markers for cleaner look
                    )[0]
                    
                    # Gradient-like fill for confidence bands
                    ax1.fill_between(
                        episodes,
                        mean_scores - std_scores,
                        mean_scores + std_scores,
                        alpha=0.18, color=color, linewidth=0,
                        edgecolor='none'
                    )

                    # Final score distribution using KDE
                    final_scores = scores[:, -1]
                    score_std = np.std(final_scores)
                    
                    # Use KDE for smooth distribution visualization
                    if SCIPY_AVAILABLE:
                        try:
                            # Calculate KDE
                            kde = stats.gaussian_kde(final_scores)
                            # Adjust bandwidth for better visualization
                            # Smaller bandwidth for concentrated data (like LIRL)
                            if score_std < 0.01:
                                kde.set_bandwidth(bw_method=kde.factor * 0.5)
                            
                            # Evaluate KDE on x grid
                            kde_values = kde(x_kde)
                            
                            # Plot KDE curve
                            ax2.plot(x_kde, kde_values, 
                                    color=color, 
                                    linewidth=2.5 if algo_name == 'LIRL' else 2,
                                    alpha=0.9 if algo_name == 'LIRL' else 0.8,
                                    label=f"{algo_name} (σ={score_std:.4f})")
                            
                            # Fill under the curve for better visualization
                            ax2.fill_between(x_kde, 0, kde_values,
                                           color=color,
                                           alpha=0.3 if algo_name == 'LIRL' else 0.25)
                            
                        except Exception as e:
                            print(f"KDE failed for {algo_name}, falling back to histogram: {e}")
                            # Fallback to histogram with normalized frequency
                            n, bins, patches = ax2.hist(
                                final_scores, 
                                bins=10, 
                                density=False,
                                weights=np.ones_like(final_scores) / len(final_scores),
                                alpha=0.65 if algo_name != 'LIRL' else 0.85,
                                color=color, 
                                edgecolor='white',
                                linewidth=1.2,
                                label=f"{algo_name} (σ={score_std:.4f})"
                            )
                    else:
                        # If scipy not available, use normalized frequency histogram
                        n, bins, patches = ax2.hist(
                            final_scores, 
                            bins=10, 
                            density=False,
                            weights=np.ones_like(final_scores) / len(final_scores),  # Normalized frequency
                            alpha=0.65 if algo_name != 'LIRL' else 0.85,
                            color=color, 
                            edgecolor='white',
                            linewidth=1.2,
                            label=f"{algo_name} (σ={score_std:.4f})"
                        )

                # Enhanced titles and labels
                ax1.set_xlabel('Training Episode', fontweight='medium', fontsize=11)
                ax1.set_ylabel('Performance Score', fontweight='medium', fontsize=11)
                ax1.set_title('Algorithm Training Performance Comparison', 
                             fontweight='bold', fontsize=14, pad=15)
                
                # Improved legend styling
                legend1 = ax1.legend(
                    title="Algorithms", 
                    loc='lower right',
                    fontsize=10,
                    title_fontsize=11,
                    frameon=True,
                    fancybox=False,  # Simple box instead of fancy
                    shadow=False,  # No shadow for cleaner look
                    borderaxespad=0.5,
                    columnspacing=1.0,
                    handlelength=2.5,
                    edgecolor='#D1D5DB',  # Light border color
                    borderpad=0.6
                )
                legend1.get_frame().set_alpha(0.95)
                legend1.get_frame().set_linewidth(0.8)
                
                _beautify_axis(ax1)

                # Set reasonable y-axis limits
                ax2.set_ylim(bottom=0)
                
                # Update labels based on visualization type
                if SCIPY_AVAILABLE:
                    ax2.set_ylabel('Probability Density (KDE)', fontweight='medium', fontsize=11)
                else:
                    ax2.set_ylabel('Normalized Frequency', fontweight='medium', fontsize=11)
                    
                ax2.set_xlabel('Final Performance Score', fontweight='medium', fontsize=11)
                ax2.set_title('Final Performance Distribution', 
                             fontweight='bold', fontsize=14, pad=15)
                
                legend2 = ax2.legend(
                    title="Algorithms",
                    loc='upper left',  # Changed from upper right to upper left
                    fontsize=10,
                    title_fontsize=11,
                    frameon=True,
                    fancybox=False,  # Simple box instead of fancy
                    shadow=False,  # No shadow for cleaner look
                    borderaxespad=0.5,
                    edgecolor='#D1D5DB',  # Light border color
                    borderpad=0.6
                )
                legend2.get_frame().set_alpha(0.95)
                legend2.get_frame().set_linewidth(0.8)
                
                _beautify_axis(ax2)

                # Add subtle annotations for better readability
                ax1.annotate('', xy=(0, 0), xytext=(0, -0.05),
                           xycoords='axes fraction', textcoords='axes fraction',
                           arrowprops=dict(arrowstyle='-', color='#E5E7EB', lw=2))
                ax2.annotate('', xy=(0, 0), xytext=(0, -0.05),
                           xycoords='axes fraction', textcoords='axes fraction',
                           arrowprops=dict(arrowstyle='-', color='#E5E7EB', lw=2))

                # Save plot with high quality and transparent background
                save_path = self.compare_dir / 'algorithm_comparison.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                          transparent=True, edgecolor='none')
                print(f"Training curve plot saved to: {save_path}")

                plt.show()

        except Exception as e:
            print(f"Plot styling failed, falling back to default style: {e}")
            # Fallback with improved colors
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            # Set transparent background for fallback
            fig.patch.set_alpha(0.0)
            ax1.patch.set_alpha(0.0)
            ax2.patch.set_alpha(0.0)
            colors = ELEGANT_PALETTE
            color_idx = 0
            
            # Collect all final scores to determine common range
            all_final_scores = []
            
            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                if scores is None or getattr(scores, "size", 0) == 0:
                    continue
                if scores.ndim != 2 or scores.shape[1] == 0:
                    continue
                final_scores = scores[:, -1]
                all_final_scores.extend(final_scores)
            
            # Determine x-axis range for visualization
            if all_final_scores:
                x_min = np.min(all_final_scores) - 0.02
                x_max = np.max(all_final_scores) + 0.02
                
                # Try KDE if scipy available
                if SCIPY_AVAILABLE:
                    x_kde = np.linspace(x_min, x_max, 200)
            
            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                if scores is None or getattr(scores, "size", 0) == 0:
                    continue
                if scores.ndim != 2 or scores.shape[1] == 0:
                    continue
                # Special color for LIRL algorithm
                if algo_name == 'LIRL':
                    color = "#EF4444"  # Modern red
                    hist_alpha = 0.85  # Higher alpha for LIRL
                else:
                    color = colors[color_idx % len(colors)]
                    color_idx += 1
                    hist_alpha = 0.65  # Lower alpha for other algorithms
                mean_scores = np.mean(scores, axis=0)
                std_scores = np.std(scores, axis=0)
                episodes = np.arange(len(mean_scores))
                ax1.plot(episodes, mean_scores, label=algo_name, color=color, linewidth=2)
                ax1.fill_between(episodes, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, color=color)
                final_scores = scores[:, -1]
                
                # Try KDE first, fallback to histogram
                if SCIPY_AVAILABLE and all_final_scores:
                    try:
                        kde = stats.gaussian_kde(final_scores)
                        kde_values = kde(x_kde)
                        ax2.plot(x_kde, kde_values, color=color, linewidth=2, 
                                alpha=hist_alpha, label=algo_name)
                        ax2.fill_between(x_kde, 0, kde_values, color=color, alpha=hist_alpha * 0.3)
                    except:
                        # Fallback to normalized frequency histogram
                        ax2.hist(final_scores, bins=15, alpha=hist_alpha, label=algo_name, 
                                color=color, edgecolor='white', linewidth=0.5, 
                                density=False, weights=np.ones_like(final_scores)/len(final_scores))
                else:
                    # Use normalized frequency histogram
                    ax2.hist(final_scores, bins=15, alpha=hist_alpha, label=algo_name, 
                            color=color, edgecolor='white', linewidth=0.5, 
                            density=False, weights=np.ones_like(final_scores)/len(final_scores))

            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score')
            ax1.set_title('Training Curve Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.set_xlabel('Final Score')
            if SCIPY_AVAILABLE:
                ax2.set_ylabel('Probability Density (KDE)')
            else:
                ax2.set_ylabel('Normalized Frequency')
            ax2.set_title('Final Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.compare_dir / 'algorithm_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"Training curve plot saved to: {save_path}")
            plt.show()
    
    def plot_convergence_speed(self):
        """Analyze and plot convergence speed comparison"""
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            print("matplotlib or numpy unavailable, skipping convergence speed plot")
            return
        
        print("\nAnalyzing convergence speed...")
        
        # Color palette
        ELEGANT_PALETTE = [
            "#6366F1",  # Indigo
            "#F59E0B",  # Amber
            "#10B981",  # Emerald
            "#EC4899",  # Pink
            "#14B8A6",  # Teal
            "#8B5CF6",  # Violet
            "#F97316",  # Orange
            "#06B6D4",  # Cyan
        ]
        
        try:
            import matplotlib.patches as mpatches
            
            # Setup figure with multiple subplots
            fig = plt.figure(figsize=(18, 10), constrained_layout=True)
            # Set transparent background
            fig.patch.set_alpha(0.0)
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
            
            ax1 = fig.add_subplot(gs[0, 0])  # Episodes to reach thresholds
            ax2 = fig.add_subplot(gs[0, 1])  # Convergence rate over time
            ax3 = fig.add_subplot(gs[1, 0])  # Performance improvement rate
            ax4 = fig.add_subplot(gs[1, 1])  # Time to stable performance
            
            # Style configuration with transparent backgrounds
            plt.rcParams.update({
                "figure.facecolor": "none",
                "axes.facecolor": "none",
                "savefig.facecolor": "none",
                "axes.grid": True,
                "grid.color": "#E5E7EB",
                "grid.alpha": 0.6,
                "axes.edgecolor": "#D1D5DB",
                "font.family": "sans-serif",
            })
            
            # Set all axes to transparent
            for ax in [ax1, ax2, ax3, ax4]:
                ax.patch.set_alpha(0.0)
            
            colors = {}
            color_idx = 0
            
            # Assign colors to algorithms
            for algo_name in self.algorithms.keys():
                if algo_name == 'LIRL':
                    colors[algo_name] = "#EF4444"  # Red for LIRL
                else:
                    colors[algo_name] = ELEGANT_PALETTE[color_idx % len(ELEGANT_PALETTE)]
                    color_idx += 1
            
            # 1. Episodes to reach performance thresholds
            print("Computing convergence thresholds...")
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]  # Performance thresholds
            threshold_data = {algo: [] for algo in self.algorithms.keys()}
            
            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                mean_scores = np.mean(scores, axis=0)
                max_score = np.max(mean_scores)
                
                for threshold in thresholds:
                    target = threshold * max_score
                    # Find first episode where mean score exceeds threshold
                    episodes_above = np.where(mean_scores >= target)[0]
                    if len(episodes_above) > 0:
                        threshold_data[algo_name].append(episodes_above[0])
                    else:
                        threshold_data[algo_name].append(len(mean_scores))
            
            # Plot bar chart for threshold reaching
            x = np.arange(len(thresholds))
            width = 0.15
            offset = 0
            
            for algo_name, episodes in threshold_data.items():
                if episodes:  # Only plot if data exists
                    ax1.bar(x + offset * width, episodes, width, 
                           label=algo_name, color=colors[algo_name], alpha=0.8)
                    offset += 1
            
            ax1.set_xlabel('Performance Threshold (% of max)', fontsize=11)
            ax1.set_ylabel('Episodes Required', fontsize=11)
            ax1.set_title('Episodes to Reach Performance Thresholds', fontweight='bold', fontsize=12)
            ax1.set_xticks(x + width * (len(self.algorithms) - 1) / 2)
            ax1.set_xticklabels([f'{int(t*100)}%' for t in thresholds])
            ax1.legend(loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # 2. Convergence rate over time (moving average of improvement)
            print("Computing convergence rates...")
            window_size = 10
            
            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                mean_scores = np.mean(scores, axis=0)
                
                # Calculate improvement rate (derivative)
                if len(mean_scores) > window_size:
                    improvement = np.diff(mean_scores)
                    # Smooth with moving average
                    smoothed_improvement = np.convolve(improvement, 
                                                      np.ones(window_size)/window_size, 
                                                      mode='valid')
                    episodes = np.arange(len(smoothed_improvement))
                    ax2.plot(episodes, smoothed_improvement, 
                            label=algo_name, color=colors[algo_name], 
                            linewidth=2, alpha=0.9)
            
            ax2.set_xlabel('Episode', fontsize=11)
            ax2.set_ylabel('Performance Improvement Rate', fontsize=11)
            ax2.set_title('Convergence Rate Over Time (Smoothed)', fontweight='bold', fontsize=12)
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # 3. Cumulative performance improvement
            print("Computing cumulative improvements...")
            
            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                mean_scores = np.mean(scores, axis=0)
                
                # Normalize to [0, 1] based on initial and final performance
                if len(mean_scores) > 0:
                    normalized = (mean_scores - mean_scores[0]) / (np.max(mean_scores) - mean_scores[0] + 1e-8)
                    episodes = np.arange(len(normalized))
                    ax3.plot(episodes, normalized * 100, 
                            label=algo_name, color=colors[algo_name], 
                            linewidth=2, alpha=0.9)
            
            ax3.set_xlabel('Episode', fontsize=11)
            ax3.set_ylabel('Performance Improvement (%)', fontsize=11)
            ax3.set_title('Relative Performance Improvement', fontweight='bold', fontsize=12)
            ax3.legend(loc='lower right', fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            # 4. Time to stable performance (box plot)
            print("Computing stability metrics...")
            stability_episodes = []
            stability_labels = []
            stability_colors = []
            
            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                episodes_to_stability = []
                
                # For each run, find when performance stabilizes
                for run_scores in scores:
                    if len(run_scores) > 20:
                        # Consider stable when std of last 20 episodes is < 5% of mean
                        for i in range(20, len(run_scores)):
                            window = run_scores[i-20:i]
                            if np.std(window) < 0.05 * np.mean(window):
                                episodes_to_stability.append(i)
                                break
                        else:
                            episodes_to_stability.append(len(run_scores))
                
                if episodes_to_stability:
                    stability_episodes.append(episodes_to_stability)
                    stability_labels.append(algo_name)
                    stability_colors.append(colors[algo_name])
            
            # Create box plot
            if stability_episodes:
                bp = ax4.boxplot(stability_episodes, labels=stability_labels, 
                                patch_artist=True, notch=True, showmeans=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], stability_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Style the plot
                for element in ['whiskers', 'fliers', 'caps']:
                    plt.setp(bp[element], color='#666666')
                plt.setp(bp['medians'], color='#333333', linewidth=2)
                plt.setp(bp['means'], marker='D', markerfacecolor='white', 
                        markeredgecolor='#333333', markersize=6)
            
            ax4.set_ylabel('Episodes to Stability', fontsize=11)
            ax4.set_title('Episodes Required to Reach Stable Performance', fontweight='bold', fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.tick_params(axis='x', rotation=45)
            
            # Overall figure title
            fig.suptitle('Convergence Speed Analysis', fontsize=16, fontweight='bold', y=1.02)
            
            # Beautify all axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#9CA3AF')
                ax.spines['bottom'].set_color('#9CA3AF')
                ax.tick_params(colors='#666666')
                ax.patch.set_alpha(0.0)  # Ensure transparent background
            
            # Save figure
            save_path = self.compare_dir / 'convergence_speed_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
            print(f"Convergence speed analysis saved to: {save_path}")
            
            plt.show()
            
            # Generate text summary
            print("\n" + "="*60)
            print("Convergence Speed Summary")
            print("="*60)
            
            # Rank algorithms by average episodes to 80% threshold
            if threshold_data:
                threshold_80_idx = thresholds.index(0.8) if 0.8 in thresholds else -1
                if threshold_80_idx >= 0:
                    rankings = [(algo, data[threshold_80_idx]) 
                               for algo, data in threshold_data.items() 
                               if len(data) > threshold_80_idx]
                    rankings.sort(key=lambda x: x[1])
                    
                    print("\nFastest to reach 80% of max performance:")
                    for i, (algo, episodes) in enumerate(rankings, 1):
                        print(f"  {i}. {algo}: {episodes} episodes")
            
            print("\n" + "-"*60)
            
        except Exception as e:
            print(f"Convergence speed analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_report(self):
        """Generate analysis report"""
        print("\n" + "="*60)
        print("Generate Analysis Report")
        print("="*60)
        
        report_lines = []
        report_lines.append("Algorithm Comparison Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Analysis directory: {self.compare_dir}")
        report_lines.append(f"Number of algorithms: {len(self.algorithms)}")
        report_lines.append("")
        
        if not NUMPY_AVAILABLE:
            report_lines.append("Note: numpy unavailable, some analysis functions were skipped")
            report_lines.append("")
        
        # Algorithm overview
        report_lines.append("Algorithm Overview:")
        for algo_name, data in self.algorithms.items():
            if NUMPY_AVAILABLE:
                scores = data['scores']
                runs = len(scores)
                episodes = len(scores[0]) if runs > 0 else 0
                report_lines.append(f"  - {algo_name}: {runs} runs, {episodes} episodes per run")
            else:
                report_lines.append(f"  - {algo_name}: data available")
        report_lines.append("")
        
        # Performance summary
        if NUMPY_AVAILABLE and self.algorithms:
            report_lines.append("Performance Summary:")
            perf_data = []
            for algo_name, data in self.algorithms.items():
                scores = data['scores']
                if scores.size > 0:
                    final_scores = scores[:, -1]
                    mean_final = np.mean(final_scores)
                    std_final = np.std(final_scores)
                    perf_data.append((mean_final, algo_name, std_final))
            
            # Sort by performance
            perf_data.sort(reverse=True)
            for i, (mean, algo, std) in enumerate(perf_data, 1):
                report_lines.append(f"  {i}. {algo}: {mean:.4f} ± {std:.4f}")
            report_lines.append("")
        
        # Save report
        report_path = self.compare_dir / 'analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Report saved to: {report_path}")
        print("\nReport Content:")
        print('\n'.join(report_lines))
        
    def run_analysis(self):
        """Run complete analysis"""
        if not self.algorithms:
            print("No algorithm results to analyze!")
            return
            
        # Run all analysis
        self.analyze_convergence()
        self.compare_algorithms()
        self.plot_training_curves()
        self.plot_convergence_speed()  # Add convergence speed analysis
        self.generate_report()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Analyze algorithm comparison results")
    parser.add_argument('--dir', type=str, help='Compare directory path')
    parser.add_argument('--plot', action='store_true', help='Only generate plots')
    parser.add_argument('--convergence', action='store_true', help='Only generate convergence speed analysis')
    parser.add_argument('--report', action='store_true', help='Only generate report')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CompareAnalyzer(args.dir)
    
    if not analyzer.algorithms:
        print("No algorithm results found, exiting...")
        return
    
    # Execute specified function
    if args.plot:
        analyzer.plot_training_curves()
    elif args.convergence:
        analyzer.plot_convergence_speed()
    elif args.report:
        analyzer.generate_report()
    else:
        analyzer.run_analysis()

if __name__ == "__main__":
    main()
