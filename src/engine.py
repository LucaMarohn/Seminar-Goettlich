import numpy as np

# -------------------------
# Game Logic
# -------------------------
def round_engine(value_cards, play, points):
    value_cards = value_cards.copy()
    play = play.copy()

    value_card_1 = value_cards[0]

    if value_card_1 >= 0:
        # highest unique wins (ties get removed iteratively)
        while len(play[play == play.max()]) > 1 and play.max() > 0:
            play[play == play.max()] = 0

        if play.max() > 0:
            i = int(np.argmax(play))
            points[0, i] += int(value_card_1)
            value_cards = value_cards[1:]
        else:
            # carry over
            value_cards = value_cards[1:]
            if len(value_cards) > 0:
                value_cards[0] += value_card_1

    else:
        # lowest unique wins (ties get removed iteratively)
        while len(play[play == play.min()]) > 1 and play.min() > 0:
            play[play == play.min()] = 0

        if len(play[play == play.min()]) < 2:
            i = int(np.argmin(play))
            points[0, i] += int(value_card_1)
            value_cards = value_cards[1:]
        else:
            # carry over
            value_cards = value_cards[1:]
            if len(value_cards) > 0:
                value_cards[0] += value_card_1

    return value_cards, points




