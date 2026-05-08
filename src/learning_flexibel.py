"""
learning_flexibel.py
====================
Deterministic comparison: coordinate-descent optimisation with
n_groups fixed to 3, 4, or 5 intervals.

For each interval count the optimizer finds the best value-card → group
assignment via coordinate descent (multiple random restarts).
The three results are then compared head-to-head.

Constraint
----------
Every interval must contain at least MIN_PER_GROUP = 3 value cards.
With 15 value cards:
    n_groups = 3  →  at least 3 cards each  (6 free to distribute)
    n_groups = 4  →  at least 3 cards each  (3 free to distribute)
    n_groups = 5  →  exactly  3 cards each  (fully constrained)

Hand-card intervals
-------------------
Group 0 maps to the lowest hand cards, group k-1 to the highest.
For group sizes s_0, …, s_{k-1}  (Σ s_i = 15):

    group 0   → hand [1,           s_0]
    group 1   → hand [s_0+1,       s_0+s_1]
    …
    group k-1 → hand [15-s_{k-1}+1, 15]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from monte_carlo import simulate_game
from strategy import (
    bot_randomizer,
    bot_negative_strategy,
    bot_interval_strategy,
    bot_get_high_cards_schonen,
    bot_memory_high_advantage,
)

# ── Constants ─────────────────────────────────────────────────────────────────

SORTED_VALUE_CARDS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_VC_TO_IDX         = {v: i for i, v in enumerate(SORTED_VALUE_CARDS)}
N_VC               = len(SORTED_VALUE_CARDS)   # 15
MAX_GROUPS         = 5
MIN_PER_GROUP      = 3
N_GROUPS_OPTS      = [3, 4, 5]

GROUP_COLORS = ['#4878CF', '#6ACC65', '#D65F5F', '#E07B2A', '#9B59B6']


def _group_names(n):
    if n == 3: return ['low', 'mid', 'high']
    if n == 4: return ['low', 'mid-low', 'mid-high', 'high']
    return ['very-low', 'low', 'mid', 'high', 'very-high']


# ── Random-strategy opponent ──────────────────────────────────────────────────

_ALL_STRATEGIES = [
    bot_randomizer,
    bot_negative_strategy,
    bot_interval_strategy,
    #bot_get_high_cards,
    bot_get_high_cards_schonen,
    bot_memory_high_advantage,
]

_strategy_counts = {s.__name__: 0 for s in _ALL_STRATEGIES}


def random_strategy_bot(player_idx, hands, value_card,
                        rng=None, reveal=False, value_cards_remaining=None):
    """Picks one of the available strategies at random on every card play."""
    if rng is None:
        rng = np.random.default_rng()
    strategy = _ALL_STRATEGIES[int(rng.integers(len(_ALL_STRATEGIES)))]
    _strategy_counts[strategy.__name__] += 1
    return strategy(player_idx, hands, value_card,
                    rng=rng, reveal=reveal,
                    value_cards_remaining=value_cards_remaining)


def print_strategy_counts(reset=False):
    total = sum(_strategy_counts.values())
    print("\nStrategy selection counts:")
    for name, count in _strategy_counts.items():
        pct = 100 * count / total if total else 0
        print(f"  {name:<30}  {count:>7}  ({pct:.1f}%)")
    print(f"  {'total':<30}  {total:>7}")
    if reset:
        for k in _strategy_counts:
            _strategy_counts[k] = 0


# ── Fixed-assignment bot ──────────────────────────────────────────────────────

class fixed_assignment_bot:
    """Bot with a fixed value-card → group assignment and fixed n_groups."""

    def __init__(self, assignment, n_groups):
        assignment = np.asarray(assignment, dtype=int)
        if len(assignment) != N_VC:
            raise ValueError(f"assignment must have length {N_VC}")
        if not (3 <= n_groups <= MAX_GROUPS):
            raise ValueError(f"n_groups must be in {N_GROUPS_OPTS}")
        counts = np.bincount(assignment, minlength=n_groups)
        if np.any(counts[:n_groups] < MIN_PER_GROUP):
            raise ValueError(
                f"Every group must contain at least {MIN_PER_GROUP} value cards.")
        self.assignment = assignment.copy()
        self.n_groups   = n_groups
        intervals       = []
        start           = 1
        for g in range(n_groups):
            size = int(np.sum(assignment == g))
            intervals.append((start, start + size - 1))
            start += size
        self._intervals = intervals

    def __call__(self, player_idx, hands, value_card,
                 rng=None, reveal=False, value_cards_remaining=None):
        if rng is None:
            rng = np.random.default_rng()
        hand   = hands[player_idx]
        vc_idx = _VC_TO_IDX.get(value_card)
        group  = (int(self.assignment[vc_idx]) if vc_idx is not None
                  else (self.n_groups - 1 if value_card > 0 else self.n_groups // 2))
        lo, hi = self._intervals[group]
        cands  = [c for c in hand if lo <= c <= hi] or list(hand)
        chosen = int(rng.choice(cands))
        hand.remove(chosen)
        if reveal:
            print(f"Bot (Player {player_idx+1}) played: {chosen}")
        return chosen


# ── Coordinate-descent optimiser ─────────────────────────────────────────────

def _evaluate(assignment, n_groups, opponents, bot_idx, n_games, rng):
    bot    = fixed_assignment_bot(assignment, n_groups)
    strats = list(opponents[:bot_idx]) + [bot] + list(opponents[bot_idx:])
    total  = sum(float(simulate_game(strats, rng)[0][bot_idx])
                 for _ in range(n_games))
    return total / n_games


def _random_valid_assignment(rng, n_groups):
    while True:
        a = rng.integers(0, n_groups, size=N_VC)
        if np.all(np.bincount(a, minlength=n_groups)[:n_groups] >= MIN_PER_GROUP):
            return a


def optimize_assignment(n_groups, opponents, bot_idx=0, n_eval=1000,
                        max_rounds=15, n_restarts=10, seed=42, verbose=True):
    """
    Coordinate-descent optimisation with n_groups fixed.

    Each restart:
      1. Draw a random valid assignment (≥ MIN_PER_GROUP cards per group).
      2. For every value card try each alternative group; accept if score improves.
      3. Repeat until no improvement or max_rounds reached.

    Returns (best_assignment, best_score, histories).
    """
    rng = np.random.default_rng(seed)

    global_best       = None
    global_best_score = -np.inf
    all_histories     = []

    for restart in range(n_restarts):
        assignment = _random_valid_assignment(rng, n_groups)
        score      = _evaluate(assignment, n_groups, opponents, bot_idx, n_eval, rng)
        history    = [(0, float(score))]

        if verbose:
            sizes = tuple(np.bincount(assignment, minlength=n_groups)[:n_groups])
            print(f"  Restart {restart+1}/{n_restarts}  sizes={sizes}  "
                  f"start score: {score:.4f}")

        for round_ in range(max_rounds):
            improved = False
            for vc in range(N_VC):
                for new_g in range(n_groups):
                    if new_g == assignment[vc]:
                        continue
                    cand = assignment.copy()
                    cand[vc] = new_g
                    if np.any(np.bincount(cand, minlength=n_groups)[:n_groups]
                              < MIN_PER_GROUP):
                        continue
                    s = _evaluate(cand, n_groups, opponents, bot_idx, n_eval, rng)
                    if s > score:
                        score      = s
                        assignment = cand
                        improved   = True

            history.append((round_ + 1, float(score)))
            if verbose:
                sizes = tuple(np.bincount(assignment, minlength=n_groups)[:n_groups])
                print(f"    Round {round_+1:2d}: score={score:.4f}  sizes={sizes}")
            if not improved:
                break

        all_histories.append(history)
        if score > global_best_score:
            global_best_score = score
            global_best       = assignment.copy()

    if verbose:
        print(f"  → Best score: {global_best_score:.4f}\n")

    return global_best, global_best_score, all_histories


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_summary(assignment, n_groups, score):
    gnames      = _group_names(n_groups)
    group_sizes = [int(np.sum(assignment == g)) for g in range(n_groups)]
    intervals   = []
    start       = 1
    for size in group_sizes:
        intervals.append((start, start + size - 1))
        start += size

    W = 70
    print("=" * W)
    print(f"n_groups = {n_groups}  –  best average score: {score:.4f}")
    print("=" * W)
    for g in range(n_groups):
        vcs    = [SORTED_VALUE_CARDS[i] for i in range(N_VC) if assignment[i] == g]
        lo, hi = intervals[g]
        print(f"  {gnames[g]:>10}  ({len(vcs):2d} cards)  {vcs}")
        print(f"              → hand [{lo}, {hi}]")
    print("=" * W + "\n")


def print_comparison_table(results_by_n):
    W = 70
    scores  = {n: results_by_n[n][1] for n in results_by_n}
    best_n  = max(scores, key=scores.get)
    print("\n" + "=" * W)
    print("Comparison  –  best score per interval count")
    print("=" * W)
    for n in sorted(results_by_n.keys()):
        marker = "  ← winner" if n == best_n else ""
        print(f"  n_groups = {n}:  avg score = {scores[n]:.4f}{marker}")
    print("=" * W + "\n")
    return best_n


def plot_comparison(results_by_n):
    """
    Figure 1: convergence curves for each n_groups (one subplot per n).
    Figure 2: final assignment bar charts stacked vertically.
    """
    n_opts = sorted(results_by_n.keys())

    # ── Figure 1: convergence ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(n_opts), figsize=(5 * len(n_opts), 4), sharey=True)
    fig.suptitle("Coordinate-descent convergence per n_groups",
                 fontsize=13, fontweight='bold')
    for ax, n in zip(axes, n_opts):
        _, _, histories = results_by_n[n]
        for r, history in enumerate(histories):
            rounds = [h[0] for h in history]
            scores = [h[1] for h in history]
            ax.plot(rounds, scores, marker='o', label=f"Restart {r+1}")
        ax.set_title(f"n = {n}", fontsize=11)
        ax.set_xlabel("Round")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Avg reward (pts)")
    plt.tight_layout()
    plt.show()

    # ── Figure 2: final assignments ───────────────────────────────────────────
    fig, axes = plt.subplots(len(n_opts), 1, figsize=(13, 3.5 * len(n_opts)))
    fig.suptitle("Best assignment per n_groups", fontsize=13, fontweight='bold')
    scores = {n: results_by_n[n][1] for n in results_by_n}
    best_n = max(scores, key=scores.get)

    for ax, n in zip(axes, n_opts):
        assignment, score, _ = results_by_n[n]
        gnames      = _group_names(n)
        group_sizes = [int(np.sum(assignment == g)) for g in range(n)]
        intervals   = []
        start       = 1
        for size in group_sizes:
            intervals.append((start, start + size - 1))
            start += size
        colors = GROUP_COLORS[:n]
        for i, vc in enumerate(SORTED_VALUE_CARDS):
            g = int(assignment[i])
            ax.bar(i, 1, color=colors[g], edgecolor='white', linewidth=1.5)
            ax.text(i, 0.5, str(vc), ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color='white')
        ax.set_xlim(-0.5, N_VC - 0.5)
        ax.set_ylim(0, 1.45)
        ax.set_xticks([]); ax.set_yticks([])
        winner_tag = "  ← winner" if n == best_n else ""
        ax.set_xlabel(f"n = {n}  |  avg score = {score:.4f}{winner_tag}", fontsize=10)
        handles = [
            Patch(facecolor=colors[g],
                  label=f'{gnames[g]}  → hand {intervals[g]}')
            for g in range(n)
        ]
        ax.legend(handles=handles, loc='upper right', ncol=n,
                  fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    plt.show()


# ── Demo run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from monte_carlo import monte_carlo_compare, print_results, plot_results

    SEED = 42

    def silent(bot):
        def wrapper(*args, **kwargs):
            kwargs['reveal'] = False
            return bot(*args, **kwargs)
        return wrapper

    opponents = [
        silent(random_strategy_bot),
        silent(random_strategy_bot),
        silent(random_strategy_bot),
    ]
    BOT_IDX = 3

    # ── Step 1: coordinate descent for each n_groups ──────────────────────────
    results_by_n = {}
    for n in N_GROUPS_OPTS:
        print("=" * 60)
        print(f"Coordinate-descent  (n_groups = {n})")
        print("=" * 60)
        assignment, score, histories = optimize_assignment(
            n_groups   = n,
            opponents  = opponents,
            bot_idx    = BOT_IDX,
            n_eval     = 1000,
            max_rounds = 15,
            n_restarts = 10,
            seed       = SEED,
            verbose    = True,
        )
        print_summary(assignment, n, score)
        results_by_n[n] = (assignment, score, histories)

    # ── Step 2: comparison ────────────────────────────────────────────────────
    best_n = print_comparison_table(results_by_n)
    plot_comparison(results_by_n)

    # ── Step 3: full Monte Carlo with all three optimised bots ────────────────
    print("=" * 60)
    print("Full Monte Carlo  (10 000 games per bot vs. same opponents)")
    print("=" * 60)

    mc_scores = {}
    for n in N_GROUPS_OPTS:
        assignment, _, _ = results_by_n[n]
        bot        = fixed_assignment_bot(assignment, n)
        strats     = opponents + [bot]
        total      = 0.0
        rng        = np.random.default_rng(SEED)
        N_MC       = 10_000
        for _ in range(N_MC):
            total += float(simulate_game(strats, rng)[0][BOT_IDX])
        mc_scores[n] = total / N_MC
        print(f"  n_groups = {n}:  MC avg score = {mc_scores[n]:.4f}")

    best_mc_n = max(mc_scores, key=mc_scores.get)
    print(f"\n  Monte Carlo winner: n_groups = {best_mc_n}  "
          f"(score = {mc_scores[best_mc_n]:.4f})\n")

    # ── Step 4: head-to-head of all three optimised bots ─────────────────────
    print("=" * 60)
    print("Head-to-head: all three optimised bots in one game")
    print("=" * 60)

    bots = [fixed_assignment_bot(results_by_n[n][0], n) for n in N_GROUPS_OPTS]
    bot_names = [f"Opt n={n}" for n in N_GROUPS_OPTS] 
    all_strats = bots 

    mc_results = monte_carlo_compare(
        bot_strategies = all_strats,
        bot_names      = bot_names,
        n_simulations  = 10_000,
        seed           = SEED,
    )
    print_results(mc_results)
    plot_results(mc_results)

    print_strategy_counts()
