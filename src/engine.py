import numpy as np
import os


# -------------------------
# Utils (macOS)
# -------------------------
def clear_screen():
    os.system("clear")


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


# -------------------------
# Player Input
# -------------------------
def get_human_choice(player_idx, hands, n_players=2, hide_for_two_players=True):
    """
    If n_players == 2 and hide_for_two_players == True:
    - player types normally
    - after confirming, screen is cleared so the other player can't see the choice

    For n_players > 2 it still works (and is usually fine), but you can disable it.
    """
    hand = hands[player_idx]

    while True:
        print("\n" + "-" * 40)
        print(f"Player {player_idx + 1}")
        print("Your hand:", hand)

        raw = input("Choose a card: ").strip()

        if not raw.isdigit():
            print("Please enter a positive integer.")
            continue

        choice = int(raw)

        if choice not in hand:
            print("That card is not in your hand.")
            continue

        hand.remove(choice)

        # Hide choice only for 2-player mode (as you requested)
        if hide_for_two_players and n_players == 2:
            input("Press ENTER and pass to the other player...")
            clear_screen()

        return choice


def bot_randomiser(player_idx, hands, rng=None, reveal=True):
    """
    Bot chooses a random card from its hand.
    reveal=True prints the bot's choice (useful for debugging / transparency).
    """
    hand = hands[player_idx]
    if rng is None:
        rng = np.random.default_rng()

    choice = int(rng.choice(hand))
    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice