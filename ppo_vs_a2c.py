"""
PPO vs A2C on CartPole-v1 — TinyReproductions Study
ECE 57000 Course Project

Reproduces the core stability claim from:
  Schulman et al. (2017). "Proximal Policy Optimization Algorithms."
  arXiv:1707.06347

Requires:
  pip install stable-baselines3 gymnasium matplotlib numpy

Usage:
  python ppo_vs_a2c.py

Output:
  - ppo_vs_a2c_multiseed.png   (shaded learning curve plot)
  - results_summary.txt        (per-seed and aggregate statistics)
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback

# ── Hyperparameters ──────────────────────────────────────────────────────────
SEEDS        = [42, 0, 1, 2, 3]      # 5 independent runs
TOTAL_STEPS  = 100_000               # timesteps per run
EVAL_FREQ    = 2_000                 # evaluate every N timesteps
EVAL_EPISODES = 10                   # episodes per evaluation point

# PPO hyperparameters (stable-baselines3 defaults for CartPole)
PPO_LR      = 3e-4
PPO_NSTEPS  = 2048

# A2C hyperparameters (stable-baselines3 defaults for CartPole)
A2C_LR      = 7e-4


# ── Callback ─────────────────────────────────────────────────────────────────
class RewardLogger(BaseCallback):
    """
    Evaluates the current policy every `eval_freq` timesteps and logs
    the mean reward over `n_eval_episodes` episodes.
    """
    def __init__(self, eval_env, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES):
        super().__init__()
        self.eval_env       = eval_env
        self.eval_freq      = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.rewards        = []
        self.timesteps      = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                episode_rewards.append(ep_reward)
            mean_reward = float(np.mean(episode_rewards))
            self.rewards.append(mean_reward)
            self.timesteps.append(self.num_timesteps)
        return True


# ── Training ─────────────────────────────────────────────────────────────────
def train_all_seeds(seeds=SEEDS):
    """
    Trains PPO and A2C agents independently for each seed.
    Returns dicts mapping seed → list of (timestep, mean_reward) pairs.
    """
    ppo_curves = {}
    a2c_curves = {}
    ts_per_seed = {}

    for seed in seeds:
        print(f"\n── Seed {seed} ──────────────────────────")
        train_env = gym.make("CartPole-v1")
        eval_env  = gym.make("CartPole-v1")

        # PPO
        ppo_cb = RewardLogger(eval_env)
        ppo_model = PPO(
            "MlpPolicy", train_env,
            learning_rate=PPO_LR,
            n_steps=PPO_NSTEPS,
            seed=seed,
            verbose=0,
        )
        ppo_model.learn(total_timesteps=TOTAL_STEPS, callback=ppo_cb)
        ppo_curves[seed]  = ppo_cb.rewards
        ts_per_seed[seed] = ppo_cb.timesteps

        ppo_final = ppo_cb.rewards[-1] if ppo_cb.rewards else float("nan")
        print(f"  PPO final reward: {ppo_final:.1f}")

        # A2C (same eval env, reset between agents)
        a2c_cb = RewardLogger(eval_env)
        a2c_model = A2C(
            "MlpPolicy", train_env,
            learning_rate=A2C_LR,
            seed=seed,
            verbose=0,
        )
        a2c_model.learn(total_timesteps=TOTAL_STEPS, callback=a2c_cb)
        a2c_curves[seed] = a2c_cb.rewards

        a2c_final = a2c_cb.rewards[-1] if a2c_cb.rewards else float("nan")
        print(f"  A2C final reward: {a2c_final:.1f}")

        train_env.close()
        eval_env.close()

    return ppo_curves, a2c_curves, ts_per_seed


# ── Aggregation ───────────────────────────────────────────────────────────────
def aggregate(curves):
    """Stack per-seed reward lists into a matrix; return mean and std per step."""
    matrix = np.array(list(curves.values()))   # shape: (n_seeds, n_eval_points)
    return matrix.mean(axis=0), matrix.std(axis=0)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_multiseed(ppo_curves, a2c_curves, timesteps, out_path="ppo_vs_a2c_multiseed.png"):
    ppo_mean, ppo_std = aggregate(ppo_curves)
    a2c_mean, a2c_std = aggregate(a2c_curves)
    ts = np.array(timesteps)

    fig, ax = plt.subplots(figsize=(9, 5))

    # PPO — solid blue
    ax.plot(ts, ppo_mean, color="#2563EB", lw=2, label="PPO (mean)")
    ax.fill_between(ts, ppo_mean - ppo_std, ppo_mean + ppo_std,
                    color="#2563EB", alpha=0.20, label="PPO ± 1 std")

    # A2C — dashed orange
    ax.plot(ts, a2c_mean, color="#EA580C", lw=2, ls="--", label="A2C (mean)")
    ax.fill_between(ts, a2c_mean - a2c_std, a2c_mean + a2c_std,
                    color="#EA580C", alpha=0.20, label="A2C ± 1 std")

    # Max reward reference line
    ax.axhline(500, color="gray", lw=1, ls=":", label="Max reward (500)")

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Mean Reward (10 episodes)", fontsize=12)
    ax.set_title("PPO vs A2C — CartPole-v1 (5 seeds)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 520)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.close()


# ── Summary ───────────────────────────────────────────────────────────────────
def print_and_save_summary(ppo_curves, a2c_curves, out_path="results_summary.txt"):
    lines = []
    lines.append("=" * 55)
    lines.append("Final-reward summary (last eval point)")
    lines.append("=" * 55)

    for seed in SEEDS:
        p = ppo_curves[seed][-1] if ppo_curves[seed] else float("nan")
        a = a2c_curves[seed][-1] if a2c_curves[seed] else float("nan")
        lines.append(f"  Seed {seed:2d}  |  PPO: {p:6.1f}  |  A2C: {a:6.1f}")

    ppo_finals = np.array([ppo_curves[s][-1] for s in SEEDS])
    a2c_finals = np.array([a2c_curves[s][-1] for s in SEEDS])
    lines.append("-" * 55)
    lines.append(
        f"  PPO  mean={ppo_finals.mean():.1f}  std={ppo_finals.std():.1f}"
        f"  min={ppo_finals.min():.1f}  max={ppo_finals.max():.1f}"
    )
    lines.append(
        f"  A2C  mean={a2c_finals.mean():.1f}  std={a2c_finals.std():.1f}"
        f"  min={a2c_finals.min():.1f}  max={a2c_finals.max():.1f}"
    )
    lines.append("=" * 55)

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(out_path, "w") as f:
        f.write(summary + "\n")
    print(f"Summary saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Training PPO and A2C on CartPole-v1 across 5 seeds...")
    ppo_curves, a2c_curves, ts_per_seed = train_all_seeds()

    # Use seed 42's timestep list as the x-axis (all seeds are identical)
    timesteps = ts_per_seed[42]

    plot_multiseed(ppo_curves, a2c_curves, timesteps)
    print_and_save_summary(ppo_curves, a2c_curves)
    print("\nDone.")
