"""
compare_n_strategies.py
=======================
Head-to-head comparison of the three optimised fixed-assignment bots
(n_groups = 3, 4, 5) found by coordinate descent in learning_flexibel.py.

The best assignments are hard-coded from the optimisation run (seed=42,
n_eval=500, n_restarts=5) so this script runs in seconds without
re-running the optimisation.

Game setup: 3-player game — one bot per interval count.
"""

import numpy as np
from monte_carlo import monte_carlo_compare, print_results, plot_results
from learning_flexibel import fixed_assignment_bot

# ── Best assignments from coordinate-descent (seed=42) ───────────────────────
#
# SORTED_VALUE_CARDS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# index               =  0   1   2   3   4  5  6  7  8  9 10 11 12 13  14
#
# n=3  score=15.91  groups: low(6)=[-1,1,2,3,6,7]  mid(6)=[-5,-4,-3,-2,4,5]  high(3)=[8,9,10]
# n=4  score=15.74  groups: low(4)=[1,3,6,7]  mid-low(3)=[-1,2,4]  mid-high(5)=[-5,-4,-3,-2,5]  high(3)=[8,9,10]
# n=5  score=9.24   groups: very-low(3)=[2,3,6]  low(3)=[-2,1,7]  mid(3)=[-5,-3,10]  high(3)=[-1,5,9]  very-high(3)=[-4,4,8]

ASSIGNMENT_N3 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 2])
ASSIGNMENT_N4 = np.array([2, 2, 2, 2, 1, 0, 1, 0, 1, 2, 0, 0, 3, 3, 3])
ASSIGNMENT_N5 = np.array([2, 4, 2, 1, 3, 1, 0, 0, 4, 3, 0, 1, 4, 3, 2])

if __name__ == "__main__":
    SEED = 42
    N_SIM = 10_000

    bot_n3 = fixed_assignment_bot(ASSIGNMENT_N3, n_groups=3)
    bot_n4 = fixed_assignment_bot(ASSIGNMENT_N4, n_groups=4)
    bot_n5 = fixed_assignment_bot(ASSIGNMENT_N5, n_groups=5)

    print("=" * 60)
    print(f"Head-to-head: n=3 vs n=4 vs n=5  ({N_SIM:,} games)")
    print("=" * 60)

    results = monte_carlo_compare(
        bot_strategies = [bot_n3, bot_n4, bot_n5],
        bot_names      = ["Opt n=3", "Opt n=4", "Opt n=5"],
        n_simulations  = N_SIM,
        seed           = SEED,
    )

    print_results(results)
    plot_results(results)
