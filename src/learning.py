"""
learning.py
===========
Implements a learning version of the interval strategy using multi-armed
bandit algorithms (ETC and Epsilon-Greedy).

Background
----------
The original `bot_interval_strategy` hard-codes a fixed assignment:
  - Value cards  6–10  →  play high hand cards (11–15)
  - Value cards  1–5   →  play low hand cards  (1–5)
  - Value cards  -5–-1 →  play mid hand cards  (6–10)

This file learns the best assignment by letting each value card
independently learn which of 3 relative groups (low / mid / high) it
belongs to. The hand-card intervals are derived from the group sizes so
that they always match:

    low  group  (n_L value cards)  →  hand cards [1,             n_L]
    mid  group  (n_M value cards)  →  hand cards [n_L+1,         n_L+n_M]
    high group  (n_H value cards)  →  hand cards [n_L+n_M+1,     15]

A value card's group assignment is NOT required to follow sorted order —
e.g. -5 can be placed in the high group (play a high hand card to
"sacrifice" it) while 10 might be placed in the low group.

Multi-Armed Bandit formulation
------------------------------
  Each of the 15 value cards has its own 3-armed bandit:
      arm 0 = place this value card in the low  group
      arm 1 = place this value card in the mid  group
      arm 2 = place this value card in the high group

  Q[vc, k] = estimated expected game reward when value card vc is
             assigned to group k.   Shape: (15, 3).

  At the start of every game, each value card independently selects a
  group. The hand-card intervals are then recomputed from the resulting
  group sizes, guaranteeing size-matching.

  Reward : total points earned by the learning bot at the end of a game.
  Q-values updated via incremental sample mean after every game.

Algorithms
----------
  ETC  – per value card, pull each of the 3 groups m times round-robin,
         then commit to argmax Q[vc] forever.

  Epsilon-Greedy – per value card, cold-start (pull each group once),
                   then ε-explore / (1-ε)-exploit independently.
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from monte_carlo import simulate_game, compute_ranks
from strategy import (
    bot_randomizer,
    bot_negative_strategy,
    bot_interval_strategy,
)

# ── Constants ────────────────────────────────────────────────────────────────

SORTED_VALUE_CARDS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_VC_TO_IDX         = {v: i for i, v in enumerate(SORTED_VALUE_CARDS)}
N_VC               = len(SORTED_VALUE_CARDS)   # 15
N_GROUPS           = 3
GROUP_NAMES        = ['low', 'mid', 'high']


# ── Learning bot ─────────────────────────────────────────────────────────────

class learning_intervals:
    """
    Callable strategy that learns, independently per value card, which of
    3 relative groups (low/mid/high) to assign it to.

    The hand-card intervals are derived from the group sizes each game:
        low  (n_L cards) → hand [1,         n_L]
        mid  (n_M cards) → hand [n_L+1,     n_L+n_M]
        high (n_H cards) → hand [n_L+n_M+1, 15]

    Usage
    -----
        bot = learning_intervals(algorithm='epsilon_greedy', epsilon=0.1)
        for game in range(n_games):
            bot.new_game()
            points, _ = simulate_game(bots, rng)
            bot.update(points[bot_player_idx])

    Parameters
    ----------
    algorithm : 'epsilon_greedy' | 'ETC'
    epsilon   : float   exploration probability (ε-greedy only)
    m         : int     exploration games per (value_card, group) pair (ETC only)
    seed      : int | None
    """

    def __init__(self, algorithm='epsilon_greedy', epsilon=0.1, m=10, seed=None):
        if algorithm not in ('epsilon_greedy', 'ETC'):
            raise ValueError("algorithm must be 'epsilon_greedy' or 'ETC'")
        if not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon must be strictly between 0 and 1")
        if m < 1:
            raise ValueError("m must be a positive integer")

        self.algorithm = algorithm
        self.epsilon   = epsilon
        self.m         = m

        # Q[vc_idx, group_idx]  and  counts[vc_idx, group_idx]
        self.Q      = np.zeros((N_VC, N_GROUPS), dtype=float)
        self.counts = np.zeros((N_VC, N_GROUPS), dtype=int)

        # ETC: committed group per value card (-1 = not yet committed)
        self._committed = np.full(N_VC, -1, dtype=int)

        # Current game state
        self.current_assignment = np.zeros(N_VC, dtype=int)  # group idx per vc
        self._hand_intervals    = [(1, 5), (6, 10), (11, 15)]  # recomputed each game
        self.current_phase      = 'explore'

        # History (one entry per game)
        self.assignment_history = []   # list of (N_VC,) arrays
        self.q_history          = []   # list of (N_VC, N_GROUPS) arrays
        self.reward_history     = []
        self.phase_history      = []

        self._rng = np.random.default_rng(seed)

    # ── Private: per-value-card group selection ──────────────────────────────

    def _choose_group_ETC(self, vc_idx):
        if self._committed[vc_idx] >= 0:
            return int(self._committed[vc_idx]), 'exploit'
        underexplored = np.where(self.counts[vc_idx] < self.m)[0]
        if len(underexplored) > 0:
            arm = int(underexplored[np.argmin(self.counts[vc_idx, underexplored])])
            return arm, 'explore'
        arm = int(np.argmax(self.Q[vc_idx]))
        self._committed[vc_idx] = arm
        return arm, 'exploit'

    def _choose_group_eg(self, vc_idx):
        never_pulled = np.where(self.counts[vc_idx] == 0)[0]
        if len(never_pulled) > 0:
            return int(never_pulled[0]), 'explore'
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, N_GROUPS)), 'explore'
        return int(np.argmax(self.Q[vc_idx])), 'exploit'

    def _ensure_min_one_per_group(self):
        """Guarantee every group contains at least one value card.

        For ETC: only moves *committed* cards so that ongoing exploration is
        never interrupted (forcing an uncommitted card away from its chosen
        arm would prevent that arm's count from ever incrementing, keeping
        the card stuck in exploration indefinitely).  Empty groups that arise
        during the exploration phase are tolerated — gameplay falls back to
        the full hand via the empty-interval guard in __call__.
        """
        group_counts = np.bincount(self.current_assignment, minlength=N_GROUPS)
        for g in range(N_GROUPS):
            if group_counts[g] == 0:
                donor_g = int(np.argmax(group_counts))
                donors  = np.where(self.current_assignment == donor_g)[0]

                if self.algorithm == 'ETC':
                    committed = [i for i in donors if self._committed[i] >= 0]
                    if not committed:
                        continue          # tolerate empty group during exploration
                    donor_idx = int(self._rng.choice(committed))
                    self._committed[donor_idx] = g   # keep commitment in sync
                else:
                    donor_idx = int(self._rng.choice(donors))

                self.current_assignment[donor_idx] = g
                group_counts[donor_g] -= 1
                group_counts[g]       += 1

    def _compute_hand_intervals(self):
        """Derive hand-card intervals from current group sizes."""
        n_low  = int(np.sum(self.current_assignment == 0))
        n_mid  = int(np.sum(self.current_assignment == 1))
        # n_high is implicitly 15 - n_low - n_mid
        low_interval  = (1,            n_low)                  if n_low > 0  else (0, 0)
        mid_interval  = (n_low + 1,    n_low + n_mid)          if n_mid > 0  else (0, 0)
        high_interval = (n_low + n_mid + 1, 15)                              # always valid
        self._hand_intervals = [low_interval, mid_interval, high_interval]

    # ── Public lifecycle ─────────────────────────────────────────────────────

    def new_game(self):
        """Select a group for every value card and compute hand intervals."""
        phases = []
        for vc_idx in range(N_VC):
            if self.algorithm == 'ETC':
                arm, phase = self._choose_group_ETC(vc_idx)
            else:
                arm, phase = self._choose_group_eg(vc_idx)
            self.current_assignment[vc_idx] = arm
            phases.append(phase)

        self._ensure_min_one_per_group()
        self._compute_hand_intervals()
        self.current_phase = 'explore' if 'explore' in phases else 'exploit'

    def update(self, reward):
        """Update Q-values for every value card's chosen group."""
        for vc_idx in range(N_VC):
            a = self.current_assignment[vc_idx]
            self.counts[vc_idx, a] += 1
            self.Q[vc_idx, a] += (reward - self.Q[vc_idx, a]) / self.counts[vc_idx, a]

        self.assignment_history.append(self.current_assignment.copy())
        self.q_history.append(self.Q.copy())
        self.reward_history.append(float(reward))
        self.phase_history.append(self.current_phase)

    def __call__(self, player_idx, hands, value_card,
                 rng=None, reveal=False, value_cards_remaining=None):
        """Strategy interface — called once per round inside a game."""
        if rng is None:
            rng = np.random.default_rng()

        hand   = hands[player_idx]
        vc_idx = _VC_TO_IDX.get(value_card)

        if vc_idx is None:
            # Accumulated pot (round_engine sums tied pots, e.g. -1+-5=-6).
            # Heuristic: positive → try to win it (high group); negative → sacrifice (mid).
            group_idx = 2 if value_card > 0 else 1
        else:
            group_idx = int(self.current_assignment[vc_idx])

        lower, upper = self._hand_intervals[group_idx]

        # Guard: interval can be empty if all value cards were assigned to other groups
        if lower > upper:
            candidates = list(hand)
        else:
            candidates = [c for c in hand if lower <= c <= upper]

        chosen = int(rng.choice(candidates if candidates else hand))
        hand.remove(chosen)

        if reveal:
            gname = GROUP_NAMES[group_idx]
            print(f"Learning bot (Player {player_idx+1}) played: {chosen}"
                  f"  [vc={value_card} → {gname}({lower}–{upper})]")

        return chosen


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_learning_simulation(bot_strategies, learning_bot_idx,
                             bot_names=None, n_simulations=10_000, seed=42):
    """Run Monte Carlo simulation with one learning_intervals bot."""
    rng          = np.random.default_rng(seed)
    n_players    = len(bot_strategies)
    learning_bot = bot_strategies[learning_bot_idx]

    if bot_names is None:
        bot_names = [f"Bot {i+1}" for i in range(n_players)]
    if len(bot_names) != n_players:
        raise ValueError("len(bot_names) must equal len(bot_strategies)")

    all_points            = np.zeros((n_simulations, n_players), dtype=float)
    all_ranks             = np.zeros((n_simulations, n_players), dtype=int)
    value_card_wins_total = np.zeros((n_players, 15), dtype=int)
    winner_set_counter    = Counter()
    rank_counts           = np.zeros((n_players, n_players), dtype=int)
    sole_win_counts       = np.zeros(n_players, dtype=int)
    shared_win_part       = np.zeros(n_players, dtype=int)
    top_finish_counts     = np.zeros(n_players, dtype=int)

    for sim in range(n_simulations):
        learning_bot.new_game()
        final_points, value_card_wins = simulate_game(bot_strategies, rng)
        learning_bot.update(float(final_points[learning_bot_idx]))

        all_points[sim]        = final_points
        value_card_wins_total += value_card_wins

        max_pts = float(np.max(final_points))
        winners = tuple(np.where(final_points == max_pts)[0])
        winner_set_counter[winners] += 1

        if len(winners) == 1:
            sole_win_counts[winners[0]] += 1
        else:
            for w in winners:
                shared_win_part[w] += 1
        for w in winners:
            top_finish_counts[w] += 1

        ranks = compute_ranks(final_points)
        all_ranks[sim] = ranks
        for i in range(n_players):
            rank_counts[i, ranks[i] - 1] += 1

    return {
        "bot_names":                bot_names,
        "n_players":                n_players,
        "n_simulations":            n_simulations,
        "all_points":               all_points,
        "all_ranks":                all_ranks,
        "avg_points":               np.mean(all_points, axis=0),
        "std_points":               np.std(all_points, axis=0),
        "avg_ranks":                np.mean(all_ranks, axis=0),
        "sole_win_counts":          sole_win_counts,
        "shared_win_participation": shared_win_part,
        "top_finish_counts":        top_finish_counts,
        "winner_set_counter":       winner_set_counter,
        "rank_counts":              rank_counts,
        "value_card_wins":          value_card_wins_total,
        # Learning-specific
        "learning_bot_idx":   learning_bot_idx,
        "algorithm":          learning_bot.algorithm,
        "assignment_history": np.array(learning_bot.assignment_history),  # (n_sims, 15)
        "q_history":          np.array(learning_bot.q_history),           # (n_sims, 15, 3)
        "reward_history":     np.array(learning_bot.reward_history),
        "phase_history":      learning_bot.phase_history,
        "final_Q":            learning_bot.Q.copy(),                      # (15, 3)
        "arm_counts":         learning_bot.counts.copy(),                 # (15, 3)
        "committed":          learning_bot._committed.copy(),             # (15,) ETC only
    }


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_learning_summary(results):
    """
    Print the learned group assignment for every value card and show
    the resulting hand-card intervals.
    """
    alg       = results["algorithm"]
    final_Q   = results["final_Q"]       # (15, 3)
    counts    = results["arm_counts"]    # (15, 3)
    committed = results["committed"]     # (15,)
    n_sims    = results["n_simulations"]

    # Derive best assignment per value card
    best_groups = np.array([
        int(committed[i]) if (alg == 'ETC' and committed[i] >= 0)
        else int(np.argmax(final_Q[i]))
        for i in range(N_VC)
    ])

    n_low  = int(np.sum(best_groups == 0))
    n_mid  = int(np.sum(best_groups == 1))
    n_high = int(np.sum(best_groups == 2))

    low_hand  = (1, n_low)
    mid_hand  = (n_low + 1, n_low + n_mid) if n_mid > 0 else (0, 0)
    high_hand = (n_low + n_mid + 1, 15)

    W = 76
    print("\n" + "=" * W)
    print(f"Learning Summary  ({alg},  {n_sims:,} games)")
    print("=" * W)
    print(f"{'Value card':>11}  {'→ Group':>8}  "
          f"{'Q low':>8}  {'Q mid':>8}  {'Q high':>8}  "
          f"{'n low':>6}  {'n mid':>6}  {'n high':>6}")
    print("-" * W)

    for i, vc in enumerate(SORTED_VALUE_CARDS):
        g    = best_groups[i]
        name = GROUP_NAMES[g]
        q    = final_Q[i]
        n    = counts[i]
        mark = " ←" if True else ""
        print(f"{vc:>11}  {name:>8}  "
              f"{q[0]:8.4f}  {q[1]:8.4f}  {q[2]:8.4f}  "
              f"{n[0]:6d}  {n[1]:6d}  {n[2]:6d}")

    print("-" * W)
    low_vcs  = [SORTED_VALUE_CARDS[i] for i in range(N_VC) if best_groups[i] == 0]
    mid_vcs  = [SORTED_VALUE_CARDS[i] for i in range(N_VC) if best_groups[i] == 1]
    high_vcs = [SORTED_VALUE_CARDS[i] for i in range(N_VC) if best_groups[i] == 2]

    print(f"\nLearned assignment:")
    print(f"  Low  group  ({n_low:2d} cards)  value cards {low_vcs}  → hand [{low_hand[0]}, {low_hand[1]}]")
    print(f"  Mid  group  ({n_mid:2d} cards)  value cards {mid_vcs}  → hand [{mid_hand[0]}, {mid_hand[1]}]")
    print(f"  High group  ({n_high:2d} cards)  value cards {high_vcs}  → hand [{high_hand[0]}, {high_hand[1]}]")

    explore_frac = results["phase_history"].count('explore') / n_sims
    print(f"\nExploration fraction: {explore_frac:.2%} of games")
    print("=" * W + "\n")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_learning_curves(results, window=500, title_suffix=""):
    """
    Three figures mirroring the original plot style.

    Figure 1 – Q-value convergence
        One line per value card showing the Q-value of its eventually-best
        group over time.  The best value card (highest final Q) is highlighted.

    Figure 2 – Group-assignment trajectory
        Two subplots (negative / positive value cards), each showing a scatter
        of (game, group chosen) coloured by phase (explore=blue, exploit=red).
        Horizontal dotted lines mark the final committed group.
        Mirrors the original s1/s2 trajectory plot.

    Figure 3 – Rolling average reward
        Smoothed reward of the learning bot (rolling mean over `window` games).
    """
    alg              = results["algorithm"]
    final_Q          = results["final_Q"]           # (15, 3)
    q_history        = results["q_history"]         # (n_sims, 15, 3)
    reward_history   = results["reward_history"]
    phase_history    = results["phase_history"]
    assignment_history = results["assignment_history"]  # (n_sims, 15)
    committed        = results["committed"]
    n_sims           = results["n_simulations"]

    best_groups = np.array([
        int(committed[i]) if (alg == 'ETC' and committed[i] >= 0)
        else int(np.argmax(final_Q[i]))
        for i in range(N_VC)
    ])

    prefix = f"{alg}" + (f"  ({title_suffix})" if title_suffix else "")
    cmap   = plt.get_cmap('tab20')
    games  = np.arange(1, n_sims + 1)

    explore_idx = np.array([i for i, p in enumerate(phase_history) if p == 'explore'])
    exploit_idx = np.array([i for i, p in enumerate(phase_history) if p == 'exploit'])

    # ── Figure 1: Q-value convergence ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"{prefix}  –  Q-value convergence (best group per value card)",
                 fontsize=13, fontweight='bold')

    best_vc = int(np.argmax([final_Q[i, best_groups[i]] for i in range(N_VC)]))

    for vc_idx, vc in enumerate(SORTED_VALUE_CARDS):
        g     = best_groups[vc_idx]
        q_seq = q_history[:, vc_idx, g]
        is_best = (vc_idx == best_vc)
        lw    = 2.5 if is_best else 1.0
        ls    = '-'  if is_best else '--'
        label = f"vc {vc}" + (" ★" if is_best else "")
        ax.plot(games, q_seq,
                color=cmap(vc_idx / (N_VC - 1)),
                lw=lw, ls=ls, label=label, alpha=0.85)

    ax.set_xlabel("Game number")
    ax.set_ylabel("Estimated avg. points  Q[vc, best group]")
    ax.legend(title="value card", ncol=5, fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ── Figure 2: Group-assignment trajectory ────────────────────────────────
    neg_indices = [i for i, v in enumerate(SORTED_VALUE_CARDS) if v < 0]   # 0–4
    pos_indices = [i for i, v in enumerate(SORTED_VALUE_CARDS) if v > 0]   # 5–14

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"{prefix}  –  Group-assignment trajectory",
                 fontsize=13, fontweight='bold')

    group_labels = {0: 'low', 1: 'mid', 2: 'high'}
    panel_data   = [
        (neg_indices, "Negative value cards (-5 … -1)"),
        (pos_indices, "Positive value cards (1 … 10)"),
    ]

    for ax, (vc_indices, panel_title) in zip(axes, panel_data):
        for vc_idx in vc_indices:
            vc     = SORTED_VALUE_CARDS[vc_idx]
            color  = cmap(vc_idx / (N_VC - 1))
            g_hist = assignment_history[:, vc_idx]   # (n_sims,) of 0/1/2

            # Slightly jitter y so overlapping dots don't stack exactly
            jitter = (vc_indices.index(vc_idx) - len(vc_indices) / 2) * 0.06

            if len(explore_idx):
                ax.scatter(explore_idx + 1,
                           g_hist[explore_idx] + jitter,
                           c='steelblue', s=3, alpha=0.25)
            if len(exploit_idx):
                ax.scatter(exploit_idx + 1,
                           g_hist[exploit_idx] + jitter,
                           c='tomato', s=3, alpha=0.25)

            # Horizontal dotted line for final committed group
            g_best = best_groups[vc_idx]
            ax.axhline(g_best + jitter, color=color, lw=1.0, ls=':',
                       label=f"vc {vc} → {group_labels[g_best]}")

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['low', 'mid', 'high'])
        ax.set_ylabel("Group chosen")
        ax.set_title(panel_title, fontsize=10)
        ax.legend(ncol=5, fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Colour legend for phase (once)
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                          markersize=6, label='explore'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
                          markersize=6, label='exploit')]
        ax.legend(handles=handles, loc='lower right', fontsize=8)

    axes[-1].set_xlabel("Game number")
    plt.tight_layout()
    plt.show()

    # ── Figure 3: Rolling average reward ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"{prefix}  –  Rolling average reward", fontsize=13, fontweight='bold')

    rewards      = np.array(reward_history, dtype=float)
    kernel       = np.ones(window) / window
    rolling_mean = np.convolve(rewards, kernel, mode='valid')
    x_rolling    = np.arange(window, n_sims + 1)

    ax.plot(x_rolling, rolling_mean, color='darkorange', lw=2,
            label=f"Rolling mean (w={window})")

    tail_mean = float(np.mean(rewards[int(0.8 * n_sims):]))
    ax.axhline(tail_mean, color='black', lw=1.5, ls='--',
               label=f"Final-20% avg: {tail_mean:.2f} pts")

    ax.set_xlabel("Game number")
    ax.set_ylabel("Points (rolling mean)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_assignment_infographic(results, save_path=None, assignment_override=None):
    """
    Four-panel infographic showing the full MAB assignment process using
    the actual learned Q-values and group assignments from `results`.

    Panel 1 – Bandit structure for 3 example value cards (real Q-values).
    Panel 2 – Resulting group assignment for all 15 value cards.
    Panel 3 – Hand-card intervals derived from group sizes.
    Panel 4 – Gameplay loop: value card → group → interval → play + Q-update.

    assignment_override : array of shape (15,) with group indices 0/1/2.
        When provided, panels 2 and 3 use this assignment instead of argmax Q.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    C_LOW  = '#4878CF'
    C_MID  = '#6ACC65'
    C_HIGH = '#D65F5F'
    C_CARD = '#F5F0E8'
    C_DARK = '#2C2C2C'
    C_GRAY = '#AAAAAA'
    GROUP_COLORS = [C_LOW, C_MID, C_HIGH]

    alg       = results["algorithm"]
    final_Q   = results["final_Q"]     # (15, 3)
    counts    = results["arm_counts"]  # (15, 3)
    committed = results["committed"]   # (15,)

    mab_groups = np.array([
        int(committed[i]) if (alg == 'ETC' and committed[i] >= 0)
        else int(np.argmax(final_Q[i]))
        for i in range(N_VC)
    ])

    # Panels 2 & 3 use the override assignment when provided (e.g. from coordinate descent)
    best_groups = np.array(assignment_override) if assignment_override is not None else mab_groups

    n_low  = int(np.sum(best_groups == 0))
    n_mid  = int(np.sum(best_groups == 1))
    n_high = int(np.sum(best_groups == 2))

    fig = plt.figure(figsize=(18, 22), facecolor='white')
    fig.suptitle(
        f"Multi-Armed Bandit: How Each Value Card Learns Its Interval  [{alg}]",
        fontsize=17, fontweight='bold', y=0.98, color=C_DARK,
    )
    gs = fig.add_gridspec(
        4, 1, height_ratios=[2.2, 1.4, 1.2, 1.4],
        hspace=0.55, top=0.95, bottom=0.03, left=0.05, right=0.97,
    )

    # ── Panel 1: bandit structure ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 18); ax1.set_ylim(0, 10); ax1.axis('off')
    ax1.set_title("① Each value card is an independent 3-armed bandit",
                  fontsize=13, fontweight='bold', color=C_DARK, pad=8, loc='left')

    example_idxs  = [0, 7, 14]   # value cards -5, 3, 10
    example_xs    = [2, 9, 16]

    for cx, vc_idx in zip(example_xs, example_idxs):
        vc_label = str(SORTED_VALUE_CARDS[vc_idx])
        card_box = FancyBboxPatch((cx - 1.1, 7.5), 2.2, 1.8,
                                  boxstyle="round,pad=0.15",
                                  facecolor=C_CARD, edgecolor=C_DARK, linewidth=2)
        ax1.add_patch(card_box)
        ax1.text(cx, 8.4, f"Value card\n{vc_label}",
                 ha='center', va='center', fontsize=11, fontweight='bold', color=C_DARK)

        arm_xs = [cx - 1.6, cx, cx + 1.6]
        for arm_i, (ax_x, gname, gc) in enumerate(zip(arm_xs, GROUP_NAMES, GROUP_COLORS)):
            is_best = (arm_i == best_groups[vc_idx])
            arm_box = FancyBboxPatch((ax_x - 0.65, 4.2), 1.3, 2.8,
                                     boxstyle="round,pad=0.1",
                                     facecolor=gc, edgecolor='white',
                                     linewidth=3 if is_best else 1,
                                     alpha=0.88 if is_best else 0.40)
            ax1.add_patch(arm_box)
            ax1.text(ax_x, 6.55, f"Arm {arm_i}\n({gname})",
                     ha='center', va='center', fontsize=8.5,
                     fontweight='bold' if is_best else 'normal', color='white')
            ax1.text(ax_x, 5.5, f"Q = {final_Q[vc_idx, arm_i]:.2f}",
                     ha='center', va='center', fontsize=8, color='white')
            ax1.text(ax_x, 4.75, f"n = {counts[vc_idx, arm_i]}",
                     ha='center', va='center', fontsize=7.5, color='white', alpha=0.9)
            if is_best:
                ax1.text(ax_x, 4.0, "▲ best", ha='center', va='center',
                         fontsize=8, fontweight='bold', color=gc)
            ax1.annotate('', xy=(ax_x, 7.05), xytext=(cx, 7.5),
                         arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2))

        ax1.text(cx, 3.3,
                 "ε-greedy: pull argmax Q\n(or random with prob ε)",
                 ha='center', va='center', fontsize=8, color=C_DARK, style='italic',
                 bbox=dict(facecolor='#EFEFEF', edgecolor=C_GRAY,
                           boxstyle='round,pad=0.3'))

    ax1.text(5.5, 6.2, '· · · 12 more value cards · · ·',
             ha='center', va='center', fontsize=10, color=C_GRAY, style='italic')

    # ── Panel 2: all 15 cards coloured by group ───────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(-0.5, 14.5); ax2.set_ylim(-0.5, 3.5); ax2.axis('off')
    ax2.set_title("② At the start of every game: each card independently selects a group",
                  fontsize=13, fontweight='bold', color=C_DARK, pad=8, loc='left')

    for i, (vc, g) in enumerate(zip(SORTED_VALUE_CARDS, best_groups)):
        card_box = FancyBboxPatch((i - 0.42, 1.2), 0.84, 1.6,
                                  boxstyle="round,pad=0.08",
                                  facecolor=GROUP_COLORS[g], edgecolor='white',
                                  linewidth=1.5, alpha=0.88)
        ax2.add_patch(card_box)
        ax2.text(i, 2.15, str(vc),
                 ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')
        ax2.text(i, 1.55, GROUP_NAMES[g][0].upper(),
                 ha='center', va='center', fontsize=8, color='white', alpha=0.9)

    for g_id, color, label in [
        (0, C_LOW,  f'Low group\n{n_low} cards'),
        (1, C_MID,  f'Mid group\n{n_mid} cards'),
        (2, C_HIGH, f'High group\n{n_high} cards'),
    ]:
        idxs = [i for i, g in enumerate(best_groups) if g == g_id]
        if not idxs:
            continue
        xmin, xmax = min(idxs) - 0.45, max(idxs) + 0.45
        ax2.plot([xmin, xmin, xmax, xmax], [1.05, 0.9, 0.9, 1.05], color=color, lw=1.8)
        ax2.text((xmin + xmax) / 2, 0.6, label,
                 ha='center', va='center', fontsize=8, color=color, fontweight='bold')

    ax2.text(7, 3.25,
             "Cards of the same color → same group  (order within a group doesn't matter)",
             ha='center', va='center', fontsize=9, color=C_DARK, style='italic')

    # ── Panel 3: interval derivation ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 18); ax3.set_ylim(0, 5); ax3.axis('off')
    ax3.set_title("③ Compute hand-card intervals from group sizes",
                  fontsize=13, fontweight='bold', color=C_DARK, pad=8, loc='left')

    interval_data = [
        (3,   C_LOW,  f'Low  ({n_low} cards)',
         f'hand [1, {n_low}]'),
        (9,   C_MID,  f'Mid  ({n_mid} cards)',
         f'hand [{n_low+1}, {n_low+n_mid}]'),
        (15,  C_HIGH, f'High ({n_high} cards)',
         f'hand [{n_low+n_mid+1}, 15]'),
    ]
    for cx, color, top_label, bot_label in interval_data:
        box = FancyBboxPatch((cx - 2.5, 0.8), 5, 3.2,
                             boxstyle="round,pad=0.2",
                             facecolor=color, edgecolor='white', linewidth=2, alpha=0.2)
        ax3.add_patch(box)
        ax3.text(cx, 3.3, top_label,
                 ha='center', va='center', fontsize=11, color=color, fontweight='bold')
        ax3.text(cx, 2.1, '↓', ha='center', va='center', fontsize=16, color=color)
        ax3.text(cx, 1.2, bot_label,
                 ha='center', va='center', fontsize=12, color=color, fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor=color,
                           boxstyle='round,pad=0.3', lw=2))

    ax3.text(9, 4.7,
             f"Group sizes: Low={n_low}, Mid={n_mid}, High={n_high}  →  "
             f"Intervals always cover all 15 hand cards, no gaps",
             ha='center', va='center', fontsize=9.5, color=C_DARK, style='italic')

    # ── Panel 4: gameplay loop + Q-update ────────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.set_xlim(0, 18); ax4.set_ylim(0, 6); ax4.axis('off')
    ax4.set_title(
        "④ During the game: use the interval — then update Q-values from the game reward",
        fontsize=13, fontweight='bold', color=C_DARK, pad=8, loc='left')

    # Example with vc=10 (index 14)
    ex_vc    = 10
    ex_g     = int(best_groups[14])
    ex_color = GROUP_COLORS[ex_g]
    ex_gname = GROUP_NAMES[ex_g]
    ex_lo    = [1, n_low+1, n_low+n_mid+1][ex_g]
    ex_hi    = [n_low, n_low+n_mid, 15][ex_g]

    steps = [
        (1.5,  C_CARD,   C_DARK,    f"Value card\n{ex_vc} revealed",      ''),
        (5.5,  ex_color, 'white',   f"Group: {ex_gname}\n(learned)",       '→'),
        (9.5,  ex_color, 'white',   f"Interval:\n[{ex_lo}, {ex_hi}]",      '→'),
        (13.5, '#E8F4E8', C_DARK,   "Play random\ncard from\ninterval",    '→'),
    ]
    for cx, fc, tc, label, arrow in steps:
        if arrow:
            ax4.annotate('', xy=(cx - 1.6, 3.5), xytext=(cx - 2.8, 3.5),
                         arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2))
        box = FancyBboxPatch((cx - 1.5, 2.0), 3.0, 3.0,
                             boxstyle="round,pad=0.2",
                             facecolor=fc, edgecolor=C_DARK, linewidth=1.5)
        ax4.add_patch(box)
        ax4.text(cx, 3.5, label,
                 ha='center', va='center', fontsize=10, color=tc, fontweight='bold')

    ax4.annotate('', xy=(16.8, 3.5), xytext=(15.2, 3.5),
                 arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2))
    update_box = FancyBboxPatch((15.3, 1.5), 2.5, 2.7,
                                boxstyle="round,pad=0.2",
                                facecolor='#FFF8E7', edgecolor='#E0A800', linewidth=2)
    ax4.add_patch(update_box)
    ax4.text(16.55, 3.6, "After game ends:",
             ha='center', va='center', fontsize=8, fontweight='bold', color='#A07000')
    ax4.text(16.55, 2.8, "Q[vc, arm] +=",
             ha='center', va='center', fontsize=8, color=C_DARK)
    ax4.text(16.55, 2.15, "(R − Q) / n",
             ha='center', va='center', fontsize=9, fontweight='bold', color='#A07000')
    ax4.text(16.55, 1.65, "incremental mean",
             ha='center', va='center', fontsize=7.5, color=C_GRAY, style='italic')

    ax4.annotate('', xy=(0.8, 0.7), xytext=(17.2, 0.7),
                 arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.8,
                                 connectionstyle='arc3,rad=-0.3'))
    ax4.text(9, 0.25,
             "Repeat for thousands of games — Q-values converge, each value card commits to its best group",
             ha='center', va='center', fontsize=9, color=C_GRAY, style='italic')

    legend_handles = [
        mpatches.Patch(facecolor=C_LOW,  label='Low  group'),
        mpatches.Patch(facecolor=C_MID,  label='Mid  group'),
        mpatches.Patch(facecolor=C_HIGH, label='High group'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved infographic to {save_path}")
    plt.show()


# ── Coordinate-descent optimizer ─────────────────────────────────────────────

class fixed_assignment_bot:
    """
    Bot that plays a fixed, pre-computed value-card → group assignment.

    The hand-card intervals are derived once from the assignment sizes:
        low  (n_L cards) → hand [1,           n_L]
        mid  (n_M cards) → hand [n_L+1,       n_L+n_M]
        high (n_H cards) → hand [n_L+n_M+1,   15]
    """

    def __init__(self, assignment):
        assignment = np.asarray(assignment, dtype=int)
        if len(assignment) != N_VC:
            raise ValueError(f"assignment must have length {N_VC}")
        if np.any(np.bincount(assignment, minlength=N_GROUPS) == 0):
            raise ValueError("Every group must contain at least one value card.")
        self.assignment = assignment.copy()
        n_L = int(np.sum(assignment == 0))
        n_M = int(np.sum(assignment == 1))
        self._intervals = [
            (1,             n_L),
            (n_L + 1,       n_L + n_M),
            (n_L + n_M + 1, 15),
        ]

    def __call__(self, player_idx, hands, value_card,
                 rng=None, reveal=False, value_cards_remaining=None):
        if rng is None:
            rng = np.random.default_rng()
        hand   = hands[player_idx]
        vc_idx = _VC_TO_IDX.get(value_card)
        group  = (int(self.assignment[vc_idx]) if vc_idx is not None
                  else (2 if value_card > 0 else 1))
        lo, hi = self._intervals[group]
        cands  = [c for c in hand if lo <= c <= hi] or list(hand)
        chosen = int(rng.choice(cands))
        hand.remove(chosen)
        if reveal:
            print(f"Bot (Player {player_idx+1}) played: {chosen}")
        return chosen


def _evaluate(assignment, opponents, bot_idx, n_games, rng):
    """Average reward for `assignment` over `n_games` simulations."""
    bot    = fixed_assignment_bot(assignment)
    strats = list(opponents[:bot_idx]) + [bot] + list(opponents[bot_idx:])
    total  = sum(float(simulate_game(strats, rng)[0][bot_idx])
                 for _ in range(n_games))
    return total / n_games


def _random_valid_assignment(rng):
    """Uniform random assignment with at least 1 card per group."""
    while True:
        a = rng.integers(0, N_GROUPS, size=N_VC)
        if np.all(np.bincount(a, minlength=N_GROUPS) >= 1):
            return a


def optimize_assignment(opponents, bot_idx=0, n_eval=500,
                        max_rounds=15, n_restarts=5, seed=42, verbose=True):
    """
    Find the best value-card → group assignment via coordinate descent.

    For each restart:
      1. Draw a random valid starting assignment.
      2. Cycle through all 15 value cards.  For each card, try the two
         alternative groups and accept the one with the highest average
         reward (estimated from `n_eval` simulated games).
      3. Repeat until no improvement in a full sweep or `max_rounds` reached.
    Return the best assignment found across all restarts.

    Parameters
    ----------
    opponents  : strategy callables that fill ALL positions except bot_idx
    bot_idx    : index where the learning bot is inserted
    n_eval     : games per evaluation  (higher → less noise, slower)
    max_rounds : max coordinate-descent rounds per restart
    n_restarts : number of independent random starts
    seed       : RNG seed
    verbose    : print per-round progress
    """
    rng = np.random.default_rng(seed)

    global_best       = None
    global_best_score = -np.inf
    all_histories     = []        # (restart, list of (round, score))

    for restart in range(n_restarts):
        assignment = _random_valid_assignment(rng)
        score      = _evaluate(assignment, opponents, bot_idx, n_eval, rng)
        history    = [(0, float(score))]

        if verbose:
            grp = np.bincount(assignment, minlength=N_GROUPS)
            print(f"\n  Restart {restart+1}/{n_restarts}  "
                  f"sizes=({grp[0]},{grp[1]},{grp[2]})  "
                  f"start score: {score:.4f}")

        for round_ in range(max_rounds):
            improved = False
            for vc in range(N_VC):
                for new_g in range(N_GROUPS):
                    if new_g == assignment[vc]:
                        continue
                    cand = assignment.copy()
                    cand[vc] = new_g
                    if np.any(np.bincount(cand, minlength=N_GROUPS) == 0):
                        continue
                    s = _evaluate(cand, opponents, bot_idx, n_eval, rng)
                    if s > score:
                        score      = s
                        assignment = cand
                        improved   = True

            history.append((round_ + 1, float(score)))
            if verbose:
                grp = np.bincount(assignment, minlength=N_GROUPS)
                print(f"    Round {round_+1:2d}: score={score:.4f}  "
                      f"sizes=({grp[0]},{grp[1]},{grp[2]})")
            if not improved:
                break

        all_histories.append(history)
        if score > global_best_score:
            global_best_score = score
            global_best       = assignment.copy()

    if verbose:
        print(f"\n  Best score across all restarts: {global_best_score:.4f}\n")

    return global_best, global_best_score, all_histories


# ── Coordinate-descent diagnostics ───────────────────────────────────────────

def print_optimization_summary(assignment, score):
    """Print the optimized group assignment and the resulting hand intervals."""
    n_L = int(np.sum(assignment == 0))
    n_M = int(np.sum(assignment == 1))
    n_H = int(np.sum(assignment == 2))
    low_hand  = (1,         n_L)
    mid_hand  = (n_L + 1,   n_L + n_M)
    high_hand = (n_L + n_M + 1, 15)

    low_vcs  = [SORTED_VALUE_CARDS[i] for i in range(N_VC) if assignment[i] == 0]
    mid_vcs  = [SORTED_VALUE_CARDS[i] for i in range(N_VC) if assignment[i] == 1]
    high_vcs = [SORTED_VALUE_CARDS[i] for i in range(N_VC) if assignment[i] == 2]

    W = 70
    print("\n" + "=" * W)
    print("Coordinate-descent  –  Optimized assignment")
    print("=" * W)
    print(f"  Best average score: {score:.4f}")
    print()
    print(f"  Low  group  ({n_L:2d} value cards)  {low_vcs}")
    print(f"               → hand cards [{low_hand[0]}, {low_hand[1]}]")
    print()
    print(f"  Mid  group  ({n_M:2d} value cards)  {mid_vcs}")
    print(f"               → hand cards [{mid_hand[0]}, {mid_hand[1]}]")
    print()
    print(f"  High group  ({n_H:2d} value cards)  {high_vcs}")
    print(f"               → hand cards [{high_hand[0]}, {high_hand[1]}]")
    print("=" * W + "\n")


def plot_optimization_results(assignment, histories, title_suffix=""):
    """
    Figure 1 – Optimization convergence
        One line per restart showing average reward improving over rounds.

    Figure 2 – Final assignment
        Each value card coloured by its assigned group (low/mid/high).
    """
    prefix = "Coordinate descent" + (f"  ({title_suffix})" if title_suffix else "")
    group_colors = ['#4878CF', '#6ACC65', '#D65F5F']   # blue / green / red

    # ── Figure 1: convergence ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"{prefix}  –  Optimization convergence",
                 fontsize=13, fontweight='bold')

    for r, history in enumerate(histories):
        rounds = [h[0] for h in history]
        scores = [h[1] for h in history]
        ax.plot(rounds, scores, marker='o', label=f"Restart {r+1}")

    ax.set_xlabel("Coordinate-descent round")
    ax.set_ylabel("Average reward (pts)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ── Figure 2: final assignment ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f"{prefix}  –  Final value-card assignment",
                 fontsize=13, fontweight='bold')

    for i, vc in enumerate(SORTED_VALUE_CARDS):
        g     = int(assignment[i])
        color = group_colors[g]
        ax.bar(i, 1, color=color, edgecolor='white', linewidth=1.5)
        ax.text(i, 0.5, str(vc), ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    ax.set_xlim(-0.5, N_VC - 0.5)
    ax.set_ylim(0, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=group_colors[0], label=f'Low  → hand [1, {int(np.sum(assignment==0))}]'),
        Patch(facecolor=group_colors[1], label=f'Mid  → hand [{int(np.sum(assignment==0))+1}, {int(np.sum(assignment==0))+int(np.sum(assignment==1))}]'),
        Patch(facecolor=group_colors[2], label=f'High → hand [{int(np.sum(assignment==0))+int(np.sum(assignment==1))+1}, 15]'),
    ]
    ax.legend(handles=legend_handles, loc='upper center',
              ncol=3, fontsize=10, framealpha=0.9)
    ax.set_title("Each bar = one value card  (left to right: −5 … 10)",
                 fontsize=9, pad=4)
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
        silent(bot_randomizer),
        silent(bot_negative_strategy),
        silent(bot_interval_strategy),
    ]
    # The learning bot is inserted at position 3 (index 3 in the 4-player game)
    BOT_IDX = 3

    # ── Step 0: run learning simulation and show infographic ─────────────────
    print("=" * 60)
    print("Learning simulation  (epsilon-greedy, 10 000 games)")
    print("=" * 60)

    eg_bot = learning_intervals(algorithm='epsilon_greedy', epsilon=0.1, seed=SEED)
    eg_strategies = opponents + [eg_bot]

    eg_results = run_learning_simulation(
        bot_strategies   = eg_strategies,
        learning_bot_idx = BOT_IDX,
        bot_names        = ["Randomizer", "Negative Strategy", "Interval", "Learning (ε-greedy)"],
        n_simulations    = 10_000,
        seed             = SEED,
    )
    print_learning_summary(eg_results)
    plot_learning_curves(eg_results, title_suffix="ε=0.1")

    # ── Step 1: coordinate descent ───────────────────────────────────────────
    print("=" * 60)
    print("Coordinate-descent optimization")
    print("=" * 60)
    print(f"  Opponents : Randomizer, Negative Strategy, Interval")
    print(f"  n_eval    : 500 games per candidate")
    print(f"  n_restarts: 5  |  max_rounds: 15")
    print()

    best_assignment, best_score, histories = optimize_assignment(
        opponents  = opponents,
        bot_idx    = BOT_IDX,
        n_eval     = 500,
        max_rounds = 15,
        n_restarts = 5,
        seed       = SEED,
        verbose    = True,
    )

    print_optimization_summary(best_assignment, best_score)
    plot_optimization_results(best_assignment, histories)
    plot_assignment_infographic(
        eg_results,
        assignment_override=best_assignment,
        save_path="mab_process.png",
    )

    # ── Step 2: full Monte Carlo comparison ──────────────────────────────────
    print("=" * 60)
    print("Full Monte Carlo comparison  (100 000 games)")
    print("=" * 60)

    optimized_bot = fixed_assignment_bot(best_assignment)
    all_strategies = opponents + [optimized_bot]
    bot_names      = ["Randomizer", "Negative Strategy",
                      "Interval", "Optimized (coord. descent)"]

    mc_results = monte_carlo_compare(
        bot_strategies = all_strategies,
        bot_names      = bot_names,
        n_simulations  = 10_000,
        seed           = SEED,
    )
    print_results(mc_results)
    plot_results(mc_results)
