import numpy as np


def bot_randomizer(player_idx, hands, value_card, rng=None, reveal=True, value_cards_remaining=None):
    hand = hands[player_idx]

    if rng is None:
        rng = np.random.default_rng()

    choice = int(rng.choice(hand))
    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice


def bot_negative_strategy(player_idx, hands, value_card, rng=None, reveal=True, value_cards_remaining=None):
    hand = hands[player_idx]

    if rng is None:
        rng = np.random.default_rng()

    negative_map = {
        -1: 4,
        -2: 6,
        -3: 8,
        -4: 9,
        -5: 10,
    }

    if value_card in negative_map and negative_map[value_card] in hand:
        choice = negative_map[value_card]
    else:
        choice = int(rng.choice(hand))

    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice


def bot_get_high_cards(player_idx, hands, value_card, rng=None, reveal=True, value_cards_remaining=None):
    hand = hands[player_idx]

    if rng is None:
        rng = np.random.default_rng()

    choice = None

    if value_cards_remaining is not None:
        positive_remaining = [v for v in value_cards_remaining if v > 0]

        if len(positive_remaining) > 0:
            highest_remaining_value_card = max(positive_remaining)

            if value_card == highest_remaining_value_card:
                choice = max(hand)

    if choice is None:
        choice = int(rng.choice(hand))

    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice

def bot_get_high_cards_schonen(player_idx, hands, value_card, rng=None, reveal=True, value_cards_remaining=None):
    hand = hands[player_idx]

    if rng is None:
        rng = np.random.default_rng()

    high_card_map = {
        10: 15,
        9: 14,
        8: 13,
        7: 12,
    }

    # Falls passende Top-Karte gezielt gespielt werden soll
    if value_card in high_card_map and high_card_map[value_card] in hand:
        choice = high_card_map[value_card]
    else:
        # Karten schützen, solange ihre Ziel-Wertkarte noch im Stapel ist
        reserved_cards = set()
        if value_cards_remaining is not None:
            for target_value, target_card in high_card_map.items():
                if target_value in value_cards_remaining and target_card in hand:
                    reserved_cards.add(target_card)

        playable_cards = [card for card in hand if card not in reserved_cards]

        if len(playable_cards) > 0:
            choice = int(rng.choice(playable_cards))
        else:
            choice = int(rng.choice(hand))

    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice

import numpy as np


def bot_interval_strategy(player_idx, hands, value_card, rng=None, reveal=True, value_cards_remaining=None):
    hand = hands[player_idx]

    if rng is None:
        rng = np.random.default_rng()

    # Für jede Wertkarte ein erlaubtes Intervall [min_card, max_card]
    interval_map = {
        10: (11, 15),
        9:  (11, 15),
        8: (11, 15),
        7: (11, 15),
        6: (11, 15),
        5: (1, 5),
        4: (1, 5),
        3: (1, 5),
        2: (1, 5),
        1: (1, 5),
        -1: (6, 10),
        -2: (6, 10),
        -3: (6, 10),
        -4: (6, 10),
        -5: (6, 10),
    }

    if value_card in interval_map:
        lower, upper = interval_map[value_card]
        candidates = [card for card in hand if lower <= card <= upper]
    else:
        candidates = []

    if len(candidates) > 0:
        choice = int(rng.choice(candidates))
    else:
        choice = int(rng.choice(hand))

    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice

def bot_interval_n3_strategy(player_idx, hands, value_card, rng=None, reveal=True, value_cards_remaining=None):
    """Optimised 3-interval strategy (ASSIGNMENT_N3 from compare_n_strategies.py).

    Groups derived from coordinate-descent (seed=42):
      Group 0 → value cards {-1,1,2,3,4,6}  → hand [1..6]
      Group 1 → value cards {-5,-4,-3,-2,5} → hand [7..11]
      Group 2 → value cards {7,8,9,10}       → hand [12..15]
    """
    hand = hands[player_idx]

    if rng is None:
        rng = np.random.default_rng()

    # value card → hand interval
    interval_map = {
        -1: (1, 6),
         1: (1, 6),
         2: (1, 6),
         3: (1, 6),
         4: (1, 6),
         6: (1, 6),
        -5: (7, 11),
        -4: (7, 11),
        -3: (7, 11),
        -2: (7, 11),
         5: (7, 11),
         7: (12, 15),
         8: (12, 15),
         9: (12, 15),
        10: (12, 15),
    }

    if value_card in interval_map:
        lo, hi = interval_map[value_card]
        candidates = [c for c in hand if lo <= c <= hi]
    else:
        candidates = []

    choice = int(rng.choice(candidates if candidates else hand))
    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice


def bot_interval_n4_strategy(player_idx, hands, value_card, rng=None, reveal=True, value_cards_remaining=None):
    """Optimised 4-interval strategy (ASSIGNMENT_N4 from compare_n_strategies.py).

    Groups derived from coordinate-descent (seed=42):
      Group 0 → value cards {-1,1,2,3,6}    → hand [1..5]
      Group 1 → value cards {-3,-2,4}        → hand [6..8]
      Group 2 → value cards {-5,-4,5,10}     → hand [9..12]
      Group 3 → value cards {7,8,9}           → hand [13..15]
    """
    hand = hands[player_idx]

    if rng is None:
        rng = np.random.default_rng()

    interval_map = {
        -1: (1, 5),
         1: (1, 5),
         2: (1, 5),
         3: (1, 5),
         6: (1, 5),
        -3: (6, 8),
        -2: (6, 8),
         4: (6, 8),
        -5: (9, 12),
        -4: (9, 12),
         5: (9, 12),
        10: (9, 12),
         7: (13, 15),
         8: (13, 15),
         9: (13, 15),
    }

    if value_card in interval_map:
        lo, hi = interval_map[value_card]
        candidates = [c for c in hand if lo <= c <= hi]
    else:
        candidates = []

    choice = int(rng.choice(candidates if candidates else hand))
    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice


# globale Zähler
highest_card_opportunity = 0
highest_card_played = 0


def bot_memory_high_advantage(player_idx, hands, value_card, rng=None, reveal=True, value_cards_remaining=None):

    global highest_card_opportunity
    global highest_card_played

    hand = hands[player_idx]

    if rng is None:
        rng = np.random.default_rng()

    my_highest = max(hand)

    opponent_cards = []
    for j, other_hand in enumerate(hands):
        if j != player_idx:
            opponent_cards.extend(other_hand)

    opponent_highest = max(opponent_cards) if len(opponent_cards) > 0 else -np.inf

    have_unique_highest = my_highest > opponent_highest

    highest_remaining_positive = None
    if value_cards_remaining is not None:
        positive_remaining = [v for v in value_cards_remaining if v > 0]
        if len(positive_remaining) > 0:
            highest_remaining_positive = max(positive_remaining)

    if have_unique_highest:
        highest_card_opportunity += 1

    if (
        have_unique_highest
        and highest_remaining_positive is not None
        and value_card == highest_remaining_positive
    ):
        choice = my_highest
        highest_card_played += 1
    else:
        choice = int(rng.choice(hand))

    hand.remove(choice)

    if reveal:
        print(f"Bot (Player {player_idx + 1}) played: {choice}")

    return choice