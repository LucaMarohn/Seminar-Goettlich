import numpy as np

n_players = 3
value_cards = np.arange(-5, 11)
value_cards = value_cards[value_cards != 0]
np.random.shuffle(value_cards)

currency_cards = np.arange(1, 16)
currency_cards = currency_cards[currency_cards != 0]

play = np.zeros((1, 3))    # money-card played by player 1, player 2, player 3
points = np.zeros((1, 3))  # points by player 1, player 2, player 3

# example of money-card played by player 1, player 2, player 3
play = np.array([[3, 3, 3]])

def round_engine(n_players, value_cards, currency_cards, play):
    value_card_1 = value_cards[0]
    if value_card_1 >= 0:
        while len(play[play == play.max()]) > 1 and play.max() > 0:
            play[play == play.max()] = 0

        if play.max() > 0:
            i = np.argmax(play)
            points[0, i] = value_card_1
            value_cards = value_cards[1:]
        else:
            value_cards = value_cards[1:]
            value_cards[0] += value_card_1

    else:
        while len(play[play == play.min()]) > 1 and play.min() > 0: 
            play[play == play.min()] = 0

        if len(play[play == play.min()]) < 2:
            i = np.argmin(play)
            points[0, i] = value_card_1
            value_cards = value_cards[1:]
        else:
            value_cards = value_cards[1:]
            value_cards[0] += value_card_1


# --- RUN ONE ROUND ---
round_engine(n_players, value_cards, currency_cards, play)

print("Value cards: ", value_cards)
print("Points: ", points)
print("Play: ", play)