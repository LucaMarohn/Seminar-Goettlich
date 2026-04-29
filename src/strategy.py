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