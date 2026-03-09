import numpy as np
from engine import round_engine
from strategy import get_human_choice, bot_randomiser, bot_negative_strategy


# -------------------------
# Player configuration
# -------------------------
n_humans = 1
n_bots = 2

# For each bot, choose exactly one strategy:
# Bot 1 -> bot_randomiser
# Bot 2 -> bot_negative_strategy
bot_strategies = [bot_randomiser, bot_negative_strategy]

# Safety check
if len(bot_strategies) != n_bots:
    raise ValueError("The number of bot strategies must equal n_bots.")

n_players = n_humans + n_bots


# -------------------------
# Game setup
# -------------------------
value_cards = np.arange(-5, 11)
value_cards = value_cards[value_cards != 0]

rng = np.random.default_rng()
rng.shuffle(value_cards)

hands = [list(range(1, 16)) for _ in range(n_players)]

# Sonderregel für 2 Spieler
if n_players == 2:
    print("2-player mode: removing 3 random value cards and 3 random hand cards per player.")

    remove_idx = rng.choice(len(value_cards), size=3, replace=False)
    value_cards = np.delete(value_cards, remove_idx)

    for i in range(n_players):
        removed_hand_cards = rng.choice(hands[i], size=3, replace=False)
        for card in removed_hand_cards:
            hands[i].remove(card)

points = np.zeros((1, n_players), dtype=int)
round_no = 1


# -------------------------
# Game loop
# -------------------------
while len(value_cards) > 0 and all(len(h) > 0 for h in hands):
    print("\n" + "-" * 40)
    print(f"Round {round_no}")
    print("Top value card:", value_cards[0])
    print("Current points:", points)

    plays = []

    for i in range(n_players):
        if i < n_humans:
            plays.append(get_human_choice(i, hands))
        else:
            bot_idx = i - n_humans
            strategy = bot_strategies[bot_idx]
            plays.append(strategy(i, hands, value_cards[0], rng))

    play = np.array(plays, dtype=int)
    print("Play vector:", play)

    points_before = points.copy()
    value_cards, points = round_engine(value_cards, play, points)

    delta = points - points_before
    if np.all(delta == 0):
        print("No unique winner -> carried into next value card.")
    else:
        winner_idx = int(np.where(delta[0] != 0)[0][0])
        print(f"Winner: Player {winner_idx + 1} (change: {int(delta[0, winner_idx])})")

    print("Points now:", points)
    print("Remaining value cards:", len(value_cards))

    round_no += 1


print("\nGame over.")
print("Final points:", points)