import numpy as np
import os

def clear_screen():
    os.system("clear")

def get_human_choice(player_idx, hands):
    """
    Ask a human player to choose a card from their hand.
    """
    hand = hands[player_idx]

    while True:
        print(f"Player {player_idx + 1}, your hand: {hand}")
        choice = input(f"Player {player_idx + 1}, choose a card: ")

        try:
            choice = int(choice)
        except ValueError:
            clear_screen()
            print("Invalid input. Please enter an integer.")
            continue

        if choice not in hand:
            clear_screen()
            print("You do not have that card. Try again.")
            continue

        hand.remove(choice)
        clear_screen()
        return choice


def bot_randomiser(player_idx, hands, value_card, rng=None, reveal=True):
    """
    Bot chooses a random card from its hand.
    The parameter value_card is included for compatibility
    with other bot strategies.
    """
    hand = hands[player_idx]

    if rng is None:
        rng = np.random.default_rng()

    choice = int(rng.choice(hand))
    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice


def bot_negative_strategy(player_idx, hands, value_card, rng=None, reveal=True):
    """
    For negative value cards, play predefined cards:
    -1 -> 5
    -2 -> 7
    -3 -> 9
    -4 -> 11
    -5 -> 12

    If the preferred card is no longer available, play random.
    For positive value cards, also play random.
    """
    hand = hands[player_idx]

    if rng is None:
        rng = np.random.default_rng()

    negative_map = {
        -1: 5,
        -2: 7,
        -3: 9,
        -4: 11,
        -5: 12,
    }

    if value_card in negative_map and negative_map[value_card] in hand:
        choice = negative_map[value_card]
    else:
        choice = int(rng.choice(hand))

    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice