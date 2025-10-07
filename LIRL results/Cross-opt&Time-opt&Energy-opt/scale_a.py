import os
import sys
import json
import math
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root on path so we can import algs/env packages
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
# Ensure the env module directory is first so ddpg_lirl_pi's `import env as ENV` resolves to env.py
ENV_DIR = os.path.join(ROOT_DIR, 'env')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if ENV_DIR in sys.path:
    sys.path.remove(ENV_DIR)
sys.path.insert(0, ENV_DIR)

from algs import ddpg_lirl_pi as alg
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_training_curve(scores, save_dir):
    # Save raw data
    np.save(os.path.join(save_dir, "scores.npy"), np.array(scores, dtype=np.float32))
    try:
        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(scores)), scores, label="Score")
        plt.title("Training Score over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curve.png"))
    finally:
        plt.close()


def extract_scheduled_tasks(env):
    tasks = []
    for job_id in range(env.num_of_jobs):
        operations = env.task_set[job_id]
        for op_id, task in enumerate(operations):
            if getattr(task, 'state', False):
                tasks.append({
                    'job_id': int(job_id),
                    'operation_id': int(op_id),
                    'robot_id': int(getattr(task, 'processing_robot', -1)),
                    'start_time': float(getattr(task, 'start_time', 0.0)),
                    'end_time': float(getattr(task, 'end_time', 0.0)),
                    'processing_time': float(getattr(task, 'processing_time', 0.0)),
                })
    tasks.sort(key=lambda x: (x['robot_id'], x['start_time']))
    return tasks


def save_gantt(tasks, env, save_dir, title="Best Episode Gantt"):
    # Save data
    with open(os.path.join(save_dir, "gantt_data.json"), "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    if not tasks:
        return

    try:
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        # Assign colors per job
        job_colors = {}
        tableau = list(plt.get_cmap('tab20').colors)
        for t in tasks:
            jid = t['job_id']
            if jid not in job_colors:
                job_colors[jid] = tableau[len(job_colors) % len(tableau)]

            ax.barh(t['robot_id'], t['processing_time'], left=t['start_time'],
                    height=0.6, color=job_colors[jid], edgecolor='black', linewidth=0.5)
            ax.text(t['start_time'] + t['processing_time']/2.0, t['robot_id'],
                    f"J{t['job_id']}.{t['operation_id']}", ha='center', va='center', fontsize=8)

        ax.set_yticks(list(range(env.num_of_robots)))
        ax.set_yticklabels([f"Robot{i}" for i in range(env.num_of_robots)])
        ax.set_xlabel('Time')
        ax.set_ylabel('Robots')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        makespan = max(t['end_time'] for t in tasks)
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(makespan, env.num_of_robots - 0.5, f'Makespan: {makespan:.2f}',
                ha='left', va='top', color='red')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "gantt.png"))
    finally:
        plt.close()


def compute_energy_analysis(env, tasks):
    # Work energy per robot by summing task energies
    work_energy = np.zeros(env.num_of_robots, dtype=np.float64)
    for t in tasks:
        r = t['robot_id']
        job_id, op_id = t['job_id'], t['operation_id']
        task_obj = env.task_set[job_id][op_id]
        exe_time = float(getattr(task_obj, 'processing_time', t['processing_time']))
        q_pickup = getattr(task_obj, 'target_position', getattr(task_obj, 'q_pickup', None))
        mass = float(getattr(task_obj, 'mass', 1.0))
        if q_pickup is None:
            # Fallback: if missing kinematic info, approximate proportional to time
            energy = exe_time
        else:
            try:
                EM_mod = getattr(alg.ENV, 'EM', None)
                if EM_mod is not None and hasattr(EM_mod, 'energy_dynamic'):
                    energy = float(EM_mod.energy_dynamic(q_pickup, mass, exe_time))
                else:
                    energy = exe_time
            except Exception:
                energy = exe_time
        work_energy[r] += energy

    # Idle energy per robot from idle times at makespan
    makespan = max((t['end_time'] for t in tasks), default=env.current_time)
    idle_stats = env.calculate_robot_idle_times(reference_time=makespan)
    idle_energy = np.zeros(env.num_of_robots, dtype=np.float64)
    for r in range(env.num_of_robots):
        idle_time = float(idle_stats.get(f'robot_{r}', {}).get('idle_time', 0.0))
        idle_energy[r] = idle_time * 5.0  # consistent with env.step assumption

    total_energy = work_energy + idle_energy
    return work_energy, idle_energy, total_energy


def save_energy_analysis(env, tasks, save_dir):
    work, idle, total = compute_energy_analysis(env, tasks)
    data = {
        'work_energy': work.tolist(),
        'idle_energy': idle.tolist(),
        'total_energy': total.tolist(),
    }
    with open(os.path.join(save_dir, "energy_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    try:
        plt.figure(figsize=(8, 5))
        idx = np.arange(env.num_of_robots)
        plt.bar(idx, work, label='Work', color='#4caf50')
        plt.bar(idx, idle, bottom=work, label='Idle', color='#ff9800')
        plt.xticks(idx, [f"R{r}" for r in range(env.num_of_robots)])
        plt.ylabel('Energy (arb.)')
        plt.title('Robot Energy Analysis (Best Episode)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "energy_analysis.png"))
    finally:
        plt.close()


def replay_best_episode(alpha, beta, action_restore_ep, num_of_jobs, num_of_robots, best_idx: int, seed: int | None = None):
    # Re-seed and advance env reset to match the episode used during training
    if seed is not None:
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    env = alg.ENV.Env(num_of_jobs, num_of_robots, alpha, beta)
    # Advance resets to the best episode index
    for _ in range(best_idx):
        env.reset()
    s = env.reset()
    done = False
    step = 0
    for a_vec in action_restore_ep:
        if done:
            break
        a_t = torch.as_tensor(a_vec, dtype=torch.float32)
        a_t = torch.clamp(a_t, 0.0, 1.0)
        action = alg.action_projection(env, a_t)
        s, r, done = env.step(action)
        step += 1
        if step > 5000:  # hard guard
            break
    tasks = extract_scheduled_tasks(env)
    return env, tasks


def run_once(alpha: float, beta: float, num_of_jobs=10, num_of_robots=3, episodes=100):
    # Output directory per spec: result/scale_a/alpha+beta
    tag = f"scale_{alpha:.1f}+{beta:.1f}"
    save_dir = os.path.join(ROOT_DIR, "result", "scale_d", tag)
    ensure_dir(save_dir)

    # Configure training
    config = alg.CONFIG.copy()
    config.update({
        'num_of_jobs': num_of_jobs,
        'num_of_robots': num_of_robots,
        'alpha': alpha,
        'beta': beta,
        'num_of_episodes': episodes,
        'enable_multi_run': False,
        'save_models': False,
        'plot_training_curve': False,
        'print_interval': max(1, episodes // 10),
    })

    # Deterministic seed per (alpha, beta)
    seed = 20250901 + int(round(alpha * 10))
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Train single run
    scores, actions, _models = alg.main(config)

    # 1) Training curve + data
    save_training_curve(scores, save_dir)

    # 2) Best-episode Gantt + data
    best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
    with open(os.path.join(save_dir, "best_episode_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            'best_episode_index': best_idx,
            'best_score': float(scores[best_idx]) if len(scores) > 0 else 0.0,
            'seed': seed
        }, f, ensure_ascii=False, indent=2)

    env_best, tasks_best = replay_best_episode(alpha, beta, actions[best_idx], num_of_jobs, num_of_robots, best_idx=best_idx, seed=seed)
    save_gantt(tasks_best, env_best, save_dir, title=f"Gantt (alpha={alpha:.1f}, beta={beta:.1f})")

    # 3) Robot energy analysis + data
    save_energy_analysis(env_best, tasks_best, save_dir)


def main():
    num_of_jobs = 1000
    num_of_robots = 5
    # Allow quick override via environment variable
    episodes_env = os.environ.get("SCALE_A_EPISODES")
    episodes = int(episodes_env) if episodes_env else 200
    alphas_env = os.environ.get("SCALE_A_ALPHAS")
    if alphas_env:
        alphas = [float(x) for x in alphas_env.split(',') if x.strip()]
    else:
        alphas = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1..0.9

    for a in alphas:
        b = round(1.0 - a, 1)
        print(f"\n=== Training for alpha={a:.1f}, beta={b:.1f} ===")
        run_once(a, b, num_of_jobs=num_of_jobs, num_of_robots=num_of_robots, episodes=episodes)
        print(f"Saved results to result/scale_a/scale_{a:.1f}+{b:.1f}")


if __name__ == "__main__":
    main()
