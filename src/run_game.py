import numpy as np
from engine import round_engine, get_human_choice, bot_randomiser

n_players = 3
value_cards = np.arange(-5, 11)
value_cards = value_cards[value_cards != 0]
np.random.shuffle(value_cards)

player_type = ["human", "bot", "bot"]
rng = np.random.default_rng()

hands = [list(range(1, 16)) for _ in range(n_players)]
points = np.zeros((1, n_players), dtype=int)  # points by player 1, player 2, player 3
round_no = 1

while len(value_cards) > 0 and all(len(h) > 0 for h in hands):
    print("\n" + "-" * 40)
    print(f"Round {round_no}")
    print("Top value card:", value_cards[0])
    print("Current points:", points)

    plays = []
    for i in range(n_players):
        if player_type[i] == "human":
            plays.append(get_human_choice(i, hands))
        else:
            plays.append(bot_randomiser(i, hands, rng))

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