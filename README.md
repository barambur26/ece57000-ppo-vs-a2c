# PPO vs A2C on CartPole-v1 — TinyReproductions Study
**ECE 57000 Course Project**

Reproduces the core stability claim from Schulman et al. (2017), *Proximal Policy Optimization Algorithms* ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347)): PPO achieves more stable and consistent learning curves than A2C on CartPole-v1, validated statistically across 5 random seeds.

---

## Results

| Algorithm | Mean Final Reward | Std | Min | Max |
|-----------|:-----------------:|:---:|:---:|:---:|
| PPO       | 500.0             | 0.0 | 500.0 | 500.0 |
| A2C       | 488.0             | 24.0 | 440.1 | 500.0 |

PPO converges to the maximum CartPole reward on every seed with zero variance. A2C reaches 500 on 4 of 5 seeds but collapses on seed 0, confirming that instability is a reproducible property of the unclipped update rule.

---

## Repository Structure

```
.
├── ppo_vs_a2c.py               # Main experiment script (written by Blas Aramburu)
├── requirements.txt            # Python dependencies
├── ppo_vs_a2c_multiseed.png    # Output: shaded learning curve plot (generated on run)
├── results_summary.txt         # Output: per-seed and aggregate statistics (generated on run)
└── README.md                   # This file
```

---

## Dependencies

- Python 3.9+
- [stable-baselines3](https://stable-baselines3.readthedocs.io/) ≥ 2.0.0
- [gymnasium](https://gymnasium.farama.org/) ≥ 0.29.0
- numpy ≥ 1.24.0
- matplotlib ≥ 3.7.0

Install all dependencies with:

```bash
pip install -r requirements.txt
```

No dataset or model downloads are required. CartPole-v1 is bundled with gymnasium.

---

## How to Run

```bash
python ppo_vs_a2c.py
```

Expected runtime: ~5–10 minutes on a standard laptop CPU (5 seeds × 2 algorithms × 100K steps).

**Outputs written to the current directory:**
- `ppo_vs_a2c_multiseed.png` — shaded mean ± 1 std learning curves for PPO and A2C
- `results_summary.txt` — per-seed final rewards and aggregate statistics

---

## Code Authorship

| File | Status |
|------|--------|
| `ppo_vs_a2c.py` | Written by Blas Aramburu for ECE 57000. The `RewardLogger` callback structure and `PPO`/`A2C` instantiation patterns follow the [stable-baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html). The multi-seed loop, aggregation logic, and shaded plotting were written from scratch. No code was copied from external repositories. |
| `requirements.txt` | Written by Blas Aramburu. |

**Lines written from scratch (not adapted):**
- `ppo_vs_a2c.py`: lines 1–60 (module docstring, constants, imports), lines 100–175 (`train_all_seeds`, `aggregate`, `plot_multiseed`, `print_and_save_summary`, `__main__`)

**Lines adapted from stable-baselines3 docs:**
- `ppo_vs_a2c.py`: lines 62–95 (`RewardLogger` callback — `__init__` and `_on_step` structure follows the official callback tutorial)

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Environment | CartPole-v1 (gymnasium) |
| Seeds | 42, 0, 1, 2, 3 |
| Total timesteps per run | 100,000 |
| Evaluation frequency | every 2,000 steps |
| Episodes per eval point | 10 |
| PPO learning rate | 3e-4 |
| PPO n_steps | 2,048 |
| A2C learning rate | 7e-4 |
| Policy network | MlpPolicy (64×64 default) |

---

## Reference

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
