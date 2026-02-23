import numpy as np

def round_engine(value_cards, play, points):
    value_cards = value_cards.copy() 
    play = play.copy()

    value_card_1 = value_cards[0]
    if value_card_1 >= 0:
        while len(play[play == play.max()]) > 1 and play.max() > 0:
            play[play == play.max()] = 0

        if play.max() > 0:
            i = np.argmax(play)
            points[0, i] += value_card_1
            value_cards = value_cards[1:]
        else:
            value_cards = value_cards[1:]
            if len(value_cards) > 0:
                value_cards[0] += value_card_1

    else:
        while len(play[play == play.min()]) > 1 and play.min() > 0: 
            play[play == play.min()] = 0

        if len(play[play == play.min()]) < 2:
            i = np.argmin(play)
            points[0, i] += value_card_1
            value_cards = value_cards[1:]
        else:
            value_cards = value_cards[1:]
            if len(value_cards) > 0:
                value_cards[0] += value_card_1
    return value_cards, points

def get_human_choice(player_idx, hands):
    hand = hands[player_idx]          # list for specific player
    while True:
        print(f"Player {player_idx+1} hand: {hand}")
        raw = input(f"Player {player_idx+1}, choose a card: ").strip()

        if not raw.isdigit():
            print("Please enter a positive integer.")
            continue

        choice = int(raw)

        if choice not in hand:
            print("That card is not in your hand.")
            continue

        hand.remove(choice)           #card can’t be used again
        return choice