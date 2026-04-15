"""
learning.py
===========
Implements a learning version of the interval strategy using multi-armed
bandit algorithms (ETC and Epsilon-Greedy).

Background
----------
The original `bot_interval_strategy` hard-codes a threshold T=6:
  - Value cards  >= 6  →  play high hand cards (11–15)
  - Value cards  1–5   →  play low hand cards  (1–5)
  - Value cards  < 0   →  play mid hand cards  (6–10)   [never learned]

This file treats T as a learnable parameter. Each possible value
T ∈ {1, …, 10} is one "arm" of a bandit. The bot plays many games, observes
its end-of-game points as a reward, and converges on the T that performs best
against the fixed set of opponents in the simulation.

Multi-Armed Bandit formulation
------------------------------
  Arms     : T ∈ {1, …, 10}   (10 arms, indexed 0–9; arm i → T = i+1)
  Reward   : total points earned by the learning bot at end of each game
  Q-value  : estimated expected reward for each arm (sample mean, updated
              incrementally after every game)

Algorithms implemented
----------------------
  ETC (Explore-Then-Commit)
    Pull each arm m times in round-robin, then commit to argmax Q forever.

  Epsilon-Greedy
    With probability ε: pull a random arm  (exploration)
    With probability 1−ε: pull argmax Q    (exploitation)
    Cold-start: any arm not yet pulled is explored first.
"""

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

# We reuse the game simulation helpers from monte_carlo.py (read-only imports)
from monte_carlo import simulate_game, compute_ranks

# Fixed-strategy bots used as opponents in the demo
from strategy import (
    bot_randomizer,
    bot_negative_strategy,
    bot_interval_strategy,
)

class learning_intervals:
    """
    A callable strategy that learns the best threshold T via bandit feedback.

    Usage within a simulation loop
    --------------------------------
        bot = learning_intervals(algorithm='epsilon_greedy', epsilon=0.1)
        for game in range(n_games):
            bot.new_game()                          # choose arm for this game
            points, _ = simulate_game(bots, rng)   # play the game
            bot.update(points[bot_player_idx])      # observe reward, update Q

    Parameters
    ----------
    algorithm : str
        'epsilon_greedy' or 'ETC'
    epsilon : float
        Exploration probability for epsilon-greedy (ignored by ETC).
    m : int
        Number of exploration games per arm for ETC (ignored by epsilon-greedy).
    n_arms : int
        Number of bandit arms (= number of candidate thresholds T).
        Arms are indexed 0..n_arms-1; arm i corresponds to T = i+1.
    seed : int or None
        Seed for the internal RNG used for arm selection.
    """

    def __init__(self, algorithm='epsilon_greedy', epsilon=0.1, m=50,
                 n_arms=10, seed=None):

        if algorithm not in ('epsilon_greedy', 'ETC'):
            raise ValueError("algorithm must be 'epsilon_greedy' or 'ETC'")
        if not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon must be strictly between 0 and 1")
        if m < 1:
            raise ValueError("m must be a positive integer")

        self.algorithm = algorithm
        self.epsilon   = epsilon
        self.m         = m
        self.n_arms    = n_arms

        self.Q      = np.zeros(n_arms, dtype=float) # Q(a)      = current estimate of E[reward | arm a]
        self.counts = np.zeros(n_arms, dtype=int)  # counts(a) = number of times arm a has been pulled so far

        self.current_arm       = None # Arm chosen for the current game
        self.current_threshold = None 
        self.current_phase     = None   # 'explore' or 'exploit'

        self._committed_arm = None # ETC-specific: the arm committed to after exploration ends, none means exploration is still ongoing

        # History (recorded after every update, used for plotting)
        self.arm_history    = []   # arm index chosen each game
        self.q_history      = []   # snapshot of Q vector after each update
        self.reward_history = []   # reward (points) received each game
        self.phase_history  = []   # 'explore' or 'exploit' each game
 
        self._rng = np.random.default_rng(seed) # Separate RNG for arm selection (not tied to the game RNG)

    
    # Private arm-selection methods
    def _choose_arm_ETC(self):
        """
        Explore-Then-Commit selection rule.

        Exploration phase  (total budget: n_arms * m games)
          Pull arms in round-robin until every arm has been pulled m times.
          Among arms with fewest pulls, pick the one with the lowest index.

        Commit phase  (all remaining games)
          At the moment exploration ends, record argmax_a Q[a] as the
          committed arm and play it exclusively from then on.  Importantly,
          the committed arm is frozen: even though Q values continue to be
          updated (for analysis purposes), the arm selection does NOT change.
          This is the defining property of classic ETC.

        Returns (arm_index, phase_label).
        """
        underexplored = np.where(self.counts < self.m)[0] # Identify arms that have not yet reached the exploration budget m

        if len(underexplored) > 0:
            arm = int(underexplored[np.argmin(self.counts[underexplored])]) #choose the arm with the fewest pulls so far
            return arm, 'explore'
        else:
            if self._committed_arm is None:
                self._committed_arm = int(np.argmax(self.Q)) # first time we enter commit phase: freeze the best arm found
            return self._committed_arm, 'exploit' # Always return the same committed arm

    def _choose_arm_epsilon_greedy(self):
        """
        Epsilon-Greedy selection rule.

        Cold-start phase
          Pull any arm that has never been pulled at all (ensures at least one
          observation per arm before exploitation can begin).

        Exploration  (probability ε)
          Pull a uniformly random arm.

        Exploitation  (probability 1 − ε)
          Pull argmax_a Q[a].

        Returns (arm_index, phase_label).
        """
        never_pulled = np.where(self.counts == 0)[0]
        if len(never_pulled) > 0:
            return int(never_pulled[0]), 'explore' # cold start: cover any arm that has never been tried

        if self._rng.random() < self.epsilon:  # ε-greedy decision
            arm = int(self._rng.integers(0, self.n_arms))
            return arm, 'explore'
        else:
            arm = int(np.argmax(self.Q))
            return arm, 'exploit'

    def new_game(self):
        """
        Select which threshold T to use for the upcoming game.

        Must be called once before each game. Sets:
          self.current_arm       : arm index chosen (0-based)
          self.current_threshold : T = current_arm + 1
          self.current_phase     : 'explore' or 'exploit'
        """
        if self.algorithm == 'ETC':
            arm, phase = self._choose_arm_ETC()
        else:
            arm, phase = self._choose_arm_epsilon_greedy()

        self.current_arm       = arm
        self.current_threshold = arm + 1   # arm index is 0-based, but T starts at 1
        self.current_phase     = phase

    def update(self, reward):
        """
        Update the Q-value estimate for the arm used in the last game.

        Uses the incremental sample-mean formula:

            Q[a]  ←  Q[a]  +  (reward − Q[a]) / count[a]

        This is algebraically equivalent to the arithmetic mean of all
        observed rewards for arm a, but computed in O(1) time and space
        (no need to store the full history of rewards).

        Parameters
        ----------
        reward : float
            Points earned by this bot at the end of the last game.
        """
        a = self.current_arm
        self.counts[a] += 1
    
        self.Q[a] += (reward - self.Q[a]) / self.counts[a] # incremental mean: move Q(a) one step toward the new observation

        # Record history for diagnostics and plotting
        self.arm_history.append(a)
        self.q_history.append(self.Q.copy())
        self.reward_history.append(float(reward))
        self.phase_history.append(self.current_phase)

    def __call__(self, player_idx, hands, value_card,
                 rng=None, reveal=False, value_cards_remaining=None):
        """
        Strategy interface — called once per round inside a game.

        Applies the threshold rule set by the last call to new_game():

            value_card >= T  (positive)  →  target interval: high cards 11–15
            0 < value_card < T           →  target interval: low cards  1–5
            value_card < 0               →  target interval: mid cards  6–10
                                            (fixed rule, not affected by T)

        If no card in the target interval remains in the hand, falls back to
        a uniformly random card from whatever is left.

        Parameters
        ----------
        player_idx : int
        hands      : list[list[int]]   all players' current hands (mutated in place)
        value_card : int               the value card revealed this round
        rng        : np.Generator      game RNG for card selection
        reveal     : bool              if True, print the card played (default False)
        value_cards_remaining : array  remaining value cards (unused here)

        Returns
        -------
        int : the hand card played
        """
        if rng is None:
            rng = np.random.default_rng()

        hand = hands[player_idx]
        T    = self.current_threshold

        # Determine target interval based on the value card and the current threshold T
        if value_card >= T: # High-value positive card: invest a strong hand card to win the points
            lower, upper = 11, 15
        elif value_card > 0:  # Low-value positive card: don't commit strong cards, play low
            lower, upper = 1, 5
        else: # negative value card: play mid cards to avoid winning negative points
            lower, upper = 6, 10

        candidates = [c for c in hand if lower <= c <= upper]   # pick a card from the target interval

        if candidates:
            chosen = int(rng.choice(candidates))
        else: # fallback if all cards in the target interval are already played
            chosen = int(rng.choice(hand))

        hand.remove(chosen)

        if reveal:
            print(f"Learning bot (Player {player_idx + 1}) played: {chosen}"
                  f"  [T={T}, target={lower}–{upper}]")

        return chosen



# Simulation runner
def run_learning_simulation(bot_strategies, learning_bot_idx,
                             bot_names=None, n_simulations=10_000, seed=42):
    """
    Run a Monte Carlo simulation where one bot is a learning_intervals agent.

    The learning bot's new_game() / update() lifecycle is called around each
    game so it accumulates reward observations and updates its Q-estimates.
    All other bots use their fixed strategies throughout.

    Parameters
    ----------
    bot_strategies     : list[callable]   one callable per player
    learning_bot_idx   : int              index of the learning_intervals bot
    bot_names          : list[str] | None player labels for output
    n_simulations      : int              total number of games to simulate
    seed               : int              RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        Standard statistics (same layout as monte_carlo_compare):
            bot_names, n_players, n_simulations,
            all_points, all_ranks, avg_points, std_points, avg_ranks,
            sole_win_counts, shared_win_participation, top_finish_counts,
            winner_set_counter, rank_counts, value_card_wins

        Learning-specific data:
            learning_bot_idx, algorithm,
            arm_history    : (n_simulations,)  arm pulled each game
            q_history      : (n_simulations, n_arms)  Q-values after each game
            reward_history : (n_simulations,)  reward observed each game
            phase_history  : list[str]         'explore' or 'exploit' each game
            final_Q        : (n_arms,)         final Q-value estimates
            arm_counts     : (n_arms,)         total pulls per arm
    """
    rng          = np.random.default_rng(seed)
    n_players    = len(bot_strategies)
    learning_bot = bot_strategies[learning_bot_idx]

    if bot_names is None:
        bot_names = [f"Bot {i+1}" for i in range(n_players)]
    if len(bot_names) != n_players:
        raise ValueError("len(bot_names) must equal len(bot_strategies)")

    # Standard accumulators
    all_points            = np.zeros((n_simulations, n_players), dtype=float)
    all_ranks             = np.zeros((n_simulations, n_players), dtype=int)
    value_card_wins_total = np.zeros((n_players, 15), dtype=int)
    winner_set_counter    = Counter()
    rank_counts           = np.zeros((n_players, n_players), dtype=int)
    sole_win_counts       = np.zeros(n_players, dtype=int)
    shared_win_part       = np.zeros(n_players, dtype=int)
    top_finish_counts     = np.zeros(n_players, dtype=int)

    for sim in range(n_simulations):

        # 1. Bandit arm selection: choose threshold T for this game
        learning_bot.new_game()

        # 2. Simulate one full game with all bots
        final_points, value_card_wins = simulate_game(bot_strategies, rng)

        # 3. Observe reward and update Q-estimate for the chosen arm
        reward = float(final_points[learning_bot_idx])
        learning_bot.update(reward)

        # --- Standard bookkeeping ---
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
        # Standard fields
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
        # Learning-specific fields
        "learning_bot_idx":  learning_bot_idx,
        "algorithm":         learning_bot.algorithm,
        "arm_history":       np.array(learning_bot.arm_history),
        "q_history":         np.array(learning_bot.q_history),
        "reward_history":    np.array(learning_bot.reward_history),
        "phase_history":     learning_bot.phase_history,
        "final_Q":           learning_bot.Q.copy(),
        "arm_counts":        learning_bot.counts.copy(),
        "n_arms":            learning_bot.n_arms,
        # For ETC: the arm frozen at the end of exploration (None for ε-greedy)
        "committed_arm":     learning_bot._committed_arm,
    }



# Diagnostics: print summary of what the learning bot learned

def print_learning_summary(results):
    """Print a concise summary of what the bandit learned."""
    alg           = results["algorithm"]
    final_Q       = results["final_Q"]
    arm_counts    = results["arm_counts"]
    n_arms        = results["n_arms"]
    committed_arm = results["committed_arm"]   # None for ε-greedy
    n_sims        = results["n_simulations"]

    # For ETC: the decision arm is the one frozen at end of exploration.
    # For ε-greedy: the "best" is argmax of the final Q estimates.
    if alg == 'ETC' and committed_arm is not None:
        display_arm = committed_arm
    else:
        display_arm = int(np.argmax(final_Q))

    display_T = display_arm + 1

    print("\n" + "=" * 60)
    print(f"Learning Summary  ({alg},  {n_sims:,} games)")
    print("=" * 60)
    print(f"{'Arm':>4}  {'T':>4}  {'Pulls':>7}  {'Q-value (avg pts)':>18}")
    print("-" * 60)
    for a in range(n_arms):
        if alg == 'ETC':
            marker = " ←  COMMITTED" if a == committed_arm else ""
        else:
            marker = " ←  BEST" if a == display_arm else ""
        print(f"{a:4d}  {a+1:4d}  {arm_counts[a]:7d}  {final_Q[a]:18.4f}{marker}")
    print("-" * 60)

    if alg == 'ETC':
        print(f"Committed threshold: T = {display_T}  (arm {display_arm})")
        print(f"  Chosen at end of exploration based on {arm_counts[display_arm] - (n_sims - results['phase_history'].count('explore'))} pull(s) during exploration.")
    else:
        print(f"Best threshold found: T = {display_T}  (arm {display_arm})")

    print(f"  → value cards >= {display_T} use high hand cards (11–15)")
    if display_T > 1:
        print(f"  → value cards 1–{display_T-1} use low hand cards (1–5)")
    print(f"  → negative value cards always use mid hand cards (6–10)")

    # Fraction of games spent exploring vs exploiting
    explore_frac = results["phase_history"].count('explore') / n_sims
    print(f"\nExploration fraction: {explore_frac:.2%} of games")
    print("=" * 60 + "\n")


# =============================================================================
# Plotting
# =============================================================================

def plot_learning_curves(results, window=200, title_suffix=""):
    """
    Three-panel figure showing the learning process.

    Panel 1 – Q-value convergence
        One line per arm (T=1..10), x-axis = game number.
        Shows how the bot's reward estimates evolve and which arms
        are ultimately judged best / worst.

    Panel 2 – Arm choice over time
        Each game is a dot: y = threshold T chosen, colour = phase.
        Makes the explore / exploit structure of the algorithm visible.

    Panel 3 – Rolling average reward
        Smoothed reward of the learning bot (rolling mean over `window` games).
        Also shows the mean reward in the final 20% of games as a dashed
        reference for the committed performance.

    Parameters
    ----------
    results : dict   output of run_learning_simulation
    window  : int    rolling-window width for the reward smoothing
    title_suffix : str  appended to the figure title (e.g. algorithm name)
    """
    arm_history    = results["arm_history"]
    q_history      = results["q_history"]
    reward_history = results["reward_history"]
    phase_history  = results["phase_history"]
    n_sims         = results["n_simulations"]
    n_arms         = results["n_arms"]
    final_Q        = results["final_Q"]
    committed_arm  = results["committed_arm"]
    alg            = results["algorithm"]
    # For ETC use the frozen committed arm; for ε-greedy use argmax of final Q
    best_arm       = committed_arm if (alg == 'ETC' and committed_arm is not None) else int(np.argmax(final_Q))
    games          = np.arange(1, n_sims + 1)

    prefix = f"{alg}" + (f"  ({title_suffix})" if title_suffix else "")
    cmap   = plt.get_cmap('tab10')

    # ------------------------------------------------------------------ #
    # Figure 1: Q-value convergence
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"{prefix}  –  Q-value convergence", fontsize=13, fontweight='bold')

    for a in range(n_arms):
        lw    = 2.5 if a == best_arm else 1.0
        ls    = '-'  if a == best_arm else '--'
        label = f"T={a+1}"
        ax.plot(games, q_history[:, a],
                color=cmap(a / n_arms), lw=lw, ls=ls, label=label, alpha=0.85)

    ax.set_xlabel("Game number")
    ax.set_ylabel("Estimated avg. points  Q[a]")
    ax.legend(ncol=5, fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    # Figure 2: Arm choice over time (coloured by phase)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"{prefix}  –  Arm choice over time", fontsize=13, fontweight='bold')

    explore_idx = [i for i, p in enumerate(phase_history) if p == 'explore']
    exploit_idx = [i for i, p in enumerate(phase_history) if p == 'exploit']

    if explore_idx:
        ax.scatter(np.array(explore_idx) + 1,
                   arm_history[explore_idx] + 1,   # +1 to show T, not arm index
                   c='steelblue', s=4, alpha=0.4, label='explore')
    if exploit_idx:
        ax.scatter(np.array(exploit_idx) + 1,
                   arm_history[exploit_idx] + 1,
                   c='tomato', s=4, alpha=0.4, label='exploit')

    ax.axhline(best_arm + 1, color='black', lw=1.5, ls=':', label=f"committed T={best_arm+1}" if alg == 'ETC' else f"best T={best_arm+1}")
    ax.set_xlabel("Game number")
    ax.set_ylabel("Threshold T chosen")
    ax.set_yticks(range(1, n_arms + 1))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------ #
    # Figure 3: Rolling average reward
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"{prefix}  –  Rolling average reward", fontsize=13, fontweight='bold')

    rewards = np.array(reward_history, dtype=float)

    # Rolling mean (valid convolution, so first window-1 points are dropped)
    kernel       = np.ones(window) / window
    rolling_mean = np.convolve(rewards, kernel, mode='valid')
    x_rolling    = np.arange(window, n_sims + 1)

    ax.plot(x_rolling, rolling_mean, color='darkorange', lw=2,
            label=f"Rolling mean (w={window})")

    # Dashed reference: mean reward in the final 20% of games
    tail_start = int(0.8 * n_sims)
    tail_mean  = float(np.mean(rewards[tail_start:]))
    ax.axhline(tail_mean, color='black', lw=1.5, ls='--',
               label=f"Final-20% avg: {tail_mean:.2f} pts")

    ax.set_xlabel("Game number")
    ax.set_ylabel("Points (rolling mean)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



# Demo run
if __name__ == "__main__":

    N_SIMS = 10_000   # total games per simulation run
    SEED   = 42

    # -----------------------------------------------------------------------
    # Helper: wrap any strategy to suppress per-round printing.
    # The existing bots default to reveal=True, which would produce
    # millions of print statements across 10k simulations.
    # -----------------------------------------------------------------------
    def silent(bot):
        """Return a version of `bot` that never prints to stdout."""
        def wrapper(*args, **kwargs):
            kwargs['reveal'] = False
            return bot(*args, **kwargs)
        return wrapper

    # -----------------------------------------------------------------------
    # Shared opponent bots (fixed strategies throughout)
    # -----------------------------------------------------------------------
    opponents = [
        silent(bot_randomizer),           # Bot 1: plays a random card every round
        silent(bot_negative_strategy),    # Bot 2: avoids negative cards strategically
        silent(bot_interval_strategy),    # Bot 3: original fixed-threshold (T=6) bot
    ]

    # -----------------------------------------------------------------------
    # Run 1: ETC (m=150 → 10 × 150 = 1500 exploration games, rest committed)
    # -----------------------------------------------------------------------
    print("Running ETC simulation …")
    etc_bot = learning_intervals(algorithm='ETC', m=150, seed=0)
    etc_bots = opponents + [etc_bot]    # learning bot is player 4 (index 3)
    etc_names = ["Randomizer", "Negative Strategy", "Interval (T=6)", "Learning (ETC)"]

    etc_results = run_learning_simulation(
        bot_strategies   = etc_bots,
        learning_bot_idx = 3,
        bot_names        = etc_names,
        n_simulations    = N_SIMS,
        seed             = SEED,
    )
    print_learning_summary(etc_results)

    # -----------------------------------------------------------------------
    # Run 2: Epsilon-Greedy (ε=0.1)
    # -----------------------------------------------------------------------
    print("Running Epsilon-Greedy simulation …")
    eps_bot = learning_intervals(algorithm='epsilon_greedy', epsilon=0.1, seed=0)
    eps_bots = opponents + [eps_bot]
    eps_names = ["Randomizer", "Negative Strategy", "Interval (T=6)", "Learning (ε-greedy)"]

    eps_results = run_learning_simulation(
        bot_strategies   = eps_bots,
        learning_bot_idx = 3,
        bot_names        = eps_names,
        n_simulations    = N_SIMS,
        seed             = SEED,
    )
    print_learning_summary(eps_results)

    # -----------------------------------------------------------------------
    # Plot learning curves for both algorithms
    # -----------------------------------------------------------------------
    plot_learning_curves(etc_results, window=200, title_suffix="m=150")
    plot_learning_curves(eps_results, window=200, title_suffix="ε=0.1")
