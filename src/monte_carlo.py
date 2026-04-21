import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from engine import round_engine
from strategy import (
    bot_randomizer,
    bot_negative_strategy,
    bot_get_high_cards,
    bot_memory_high_advantage, 
    bot_get_high_cards_schonen,
    bot_interval_strategy
)

from strategy import highest_card_opportunity, highest_card_played

def simulate_game(bot_strategies, rng):
    n_players = len(bot_strategies)

    # Originaler Kartenstapel: nur diese Karten sollen im Tracking gezählt werden
    original_value_cards = np.arange(-5, 11)
    original_value_cards = original_value_cards[original_value_cards != 0]
    rng.shuffle(original_value_cards)

    # Arbeitskopie für das eigentliche Spiel
    value_cards = original_value_cards.copy()

    hands = [list(range(1, 16)) for _ in range(n_players)]

    # Die Einzelkarten, die am Ende geplottet werden sollen
    value_card_labels = [-5, -4, -3, -2, -1] + list(range(1, 11))
    value_to_index = {v: i for i, v in enumerate(value_card_labels)}
    value_card_wins = np.zeros((n_players, len(value_card_labels)), dtype=int)

    # Hier sammeln wir die tatsächlich aufgedeckten Einzelkarten,
    # die aktuell noch im nicht entschiedenen Pot liegen
    pending_value_cards = []

    # Sonderregel für 2 Spieler
    if n_players == 2:
        remove_idx = rng.choice(len(value_cards), size=3, replace=False)
        remove_idx = np.sort(remove_idx)

        # Dieselben Karten aus Originalstapel und Arbeitsstapel entfernen
        original_value_cards = np.delete(original_value_cards, remove_idx)
        value_cards = np.delete(value_cards, remove_idx)

        for i in range(n_players):
            removed_hand_cards = rng.choice(hands[i], size=3, replace=False)
            for card in removed_hand_cards:
                hands[i].remove(card)

    points = np.zeros((1, n_players), dtype=int)

    # Zeiger darauf, welche originale Einzelkarte als Nächstes aufgedeckt wird
    reveal_index = 0

    while len(value_cards) > 0 and all(len(h) > 0 for h in hands):
        # Tatsächlich neu aufgedeckte Einzelkarte
        revealed_single_card = int(original_value_cards[reveal_index])
        reveal_index += 1

        # Aktuell sichtbarer Potwert, evtl. schon kumuliert
        visible_value_card = int(value_cards[0])

        pending_value_cards.append(revealed_single_card)

        points_before = points.copy()
        plays = []

        for i, strategy in enumerate(bot_strategies):
            chosen_card = strategy(
                i,
                hands,
                visible_value_card,
                rng,
                value_cards_remaining=value_cards.copy(),
            )
            plays.append(chosen_card)

        play = np.array(plays, dtype=int)
        value_cards, points = round_engine(value_cards, play, points)

        delta = points - points_before
        round_winners = np.where(delta[0] != 0)[0]

        # Wenn genau ein Spieler Punkte bekam, war der ganze offene Pot entschieden
        if len(round_winners) == 1:
            winner_idx = int(round_winners[0])

            for card in pending_value_cards:
                value_card_wins[winner_idx, value_to_index[card]] += 1

            pending_value_cards = []

    return points.flatten(), value_card_wins


def compute_ranks(points):
    n = len(points)
    ranks = np.zeros(n, dtype=int)

    unique_scores = sorted(set(points), reverse=True)
    current_rank = 1

    for score in unique_scores:
        idx = np.where(points == score)[0]
        ranks[idx] = current_rank
        current_rank += len(idx)

    return ranks


def monte_carlo_compare(bot_strategies, bot_names=None, n_simulations=10000, seed=42):
    rng = np.random.default_rng(seed)
    n_players = len(bot_strategies)

    if bot_names is None:
        bot_names = [f"Bot {i + 1}" for i in range(n_players)]

    if len(bot_names) != n_players:
        raise ValueError("len(bot_names) must equal len(bot_strategies).")

    all_points = np.zeros((n_simulations, n_players), dtype=float)
    all_ranks = np.zeros((n_simulations, n_players), dtype=int)

    sole_win_counts = np.zeros(n_players, dtype=int)
    shared_win_participation = np.zeros(n_players, dtype=int)
    top_finish_counts = np.zeros(n_players, dtype=int)

    winner_set_counter = Counter()
    rank_counts = np.zeros((n_players, n_players), dtype=int)

    value_card_wins_total = np.zeros((n_players, 15), dtype=int)

    for sim in range(n_simulations):
        final_points, value_card_wins = simulate_game(bot_strategies, rng)
        all_points[sim] = final_points
        value_card_wins_total += value_card_wins

        max_points = np.max(final_points)
        winners = tuple(np.where(final_points == max_points)[0])
        winner_set_counter[winners] += 1

        if len(winners) == 1:
            sole_win_counts[winners[0]] += 1
        else:
            for w in winners:
                shared_win_participation[w] += 1

        for w in winners:
            top_finish_counts[w] += 1

        ranks = compute_ranks(final_points)
        all_ranks[sim] = ranks

        for i in range(n_players):
            rank_counts[i, ranks[i] - 1] += 1

    avg_points = np.mean(all_points, axis=0)
    std_points = np.std(all_points, axis=0)
    avg_ranks = np.mean(all_ranks, axis=0)

    results = {
        "bot_names": bot_names,
        "n_players": n_players,
        "n_simulations": n_simulations,
        "all_points": all_points,
        "all_ranks": all_ranks,
        "avg_points": avg_points,
        "std_points": std_points,
        "avg_ranks": avg_ranks,
        "sole_win_counts": sole_win_counts,
        "shared_win_participation": shared_win_participation,
        "top_finish_counts": top_finish_counts,
        "winner_set_counter": winner_set_counter,
        "rank_counts": rank_counts,
        "value_card_wins": value_card_wins_total,
    }

    return results


def print_results(results):
    bot_names = results["bot_names"]
    n_players = results["n_players"]
    n_simulations = results["n_simulations"]

    avg_points = results["avg_points"]
    std_points = results["std_points"]
    avg_ranks = results["avg_ranks"]
    sole_win_counts = results["sole_win_counts"]
    shared_win_participation = results["shared_win_participation"]
    top_finish_counts = results["top_finish_counts"]
    winner_set_counter = results["winner_set_counter"]
    rank_counts = results["rank_counts"]
    value_card_wins = results["value_card_wins"]

    value_card_labels = [-5, -4, -3, -2, -1] + list(range(1, 11))

    print("\n" + "=" * 70)
    print(f"Monte-Carlo comparison over {n_simulations} simulated games")
    print("=" * 70)

    print("\nPer-player statistics")
    print("-" * 70)

    for i, name in enumerate(bot_names):
        print(f"\n{name}")
        print(f"  Sole wins:                {sole_win_counts[i]:6d}   ({sole_win_counts[i] / n_simulations:.4f})")
        print(f"  Shared-win participation: {shared_win_participation[i]:6d}   ({shared_win_participation[i] / n_simulations:.4f})")
        print(f"  Top finishes total:       {top_finish_counts[i]:6d}   ({top_finish_counts[i] / n_simulations:.4f})")
        print(f"  Average points:           {avg_points[i]:8.4f}")
        print(f"  Std. dev. points:         {std_points[i]:8.4f}")
        print(f"  Average rank:             {avg_ranks[i]:8.4f}")

        rank_line = ", ".join(
            f"Rank {r + 1}: {rank_counts[i, r]} ({rank_counts[i, r] / n_simulations:.4f})"
            for r in range(n_players)
        )
        print(f"  Rank distribution:        {rank_line}")

        card_line = ", ".join(
            f"{card}: {value_card_wins[i, j]}"
            for j, card in enumerate(value_card_labels)
        )
        print(f"  Won single value cards:   {card_line}")

    print("\nWinner-set statistics")
    print("-" * 70)

    sorted_winner_sets = sorted(
        winner_set_counter.items(),
        key=lambda x: (len(x[0]), x[0])
    )

    for winners, count in sorted_winner_sets:
        label = ", ".join(bot_names[i] for i in winners)
        if len(winners) == 1:
            prefix = "Sole winner"
        else:
            prefix = f"Shared win ({len(winners)} players)"
        print(f"{prefix:<24} [{label}] : {count:6d} ({count / n_simulations:.4f})")

    if n_players == 2:
        diff = results["all_points"][:, 0] - results["all_points"][:, 1]
        print("\nTwo-player extra statistic")
        print("-" * 70)
        print(f"Average point difference ({bot_names[0]} - {bot_names[1]}): {np.mean(diff):.4f}")


def plot_results(results):
    bot_names = results["bot_names"]
    n_players = results["n_players"]

    avg_points = results["avg_points"]
    std_points = results["std_points"]
    rank_counts = results["rank_counts"]
    all_points = results["all_points"]
    value_card_wins = results["value_card_wins"]

    value_card_labels = [-5, -4, -3, -2, -1] + list(range(1, 11))

    # 1) Durchschnittliche Punkte
    plt.figure(figsize=(8, 5))
    plt.bar(bot_names, avg_points, yerr=std_points, capsize=5)
    plt.ylabel("Average points")
    plt.title("Average points with standard deviation")
    plt.tight_layout()

    # 2) Rangverteilung
    plt.figure(figsize=(10, 6))
    x = np.arange(n_players)
    width = 0.6 / n_players

    for r in range(n_players):
        rank_rate = rank_counts[:, r] / results["n_simulations"]
        plt.bar(
            x + r * width - (n_players - 1) * width / 2,
            rank_rate,
            width,
            label=f"Rank {r + 1}"
        )

    plt.xticks(x, bot_names, rotation=15)
    plt.ylabel("Frequency")
    plt.title("Rank distribution")
    plt.legend()
    plt.tight_layout()

    # 3) Verteilung der Endpunkte
    plt.figure(figsize=(10, 6))
    min_point = int(np.min(all_points))
    max_point = int(np.max(all_points))
    point_values = np.arange(min_point, max_point + 1)
    width = 0.8 / n_players

    for i in range(n_players):
        player_points = all_points[:, i]
        counts = np.array([np.sum(player_points == p) for p in point_values])

        plt.bar(
            point_values + i * width - (n_players - 1) * width / 2,
            counts,
            width,
            label=bot_names[i]
        )

    plt.xlabel("Final points")
    plt.ylabel("Number of occurrences")
    plt.title("Point frequency distribution")
    plt.legend()
    plt.tight_layout()

    # 4) Einzelne Wertkarten
    plt.figure(figsize=(11, 6))
    x = np.arange(len(value_card_labels))
    width = 0.8 / n_players

    for i in range(n_players):
        plt.bar(
            x + i * width - (n_players - 1) * width / 2,
            value_card_wins[i],
            width,
            label=bot_names[i]
        )

    plt.xticks(x, value_card_labels)
    plt.xlabel("Single value card")
    plt.ylabel("Number of times awarded")
    plt.title("Single value cards awarded by strategy")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    bot_strategies = [
        bot_randomizer,
        #bot_negative_strategy,
        #bot_get_high_cards,
        #bot_memory_high_advantage, 
        bot_get_high_cards_schonen,
        bot_get_high_cards_schonen,
        bot_interval_strategy,
        bot_interval_strategy
    ]

    bot_names = [
        "Randomizer A",
        #"Negative Strategy",
        #"Get High Cards",
        #Memory High Advantage",
        "Get High Cards",
        "Get High Cards 2",
        "Interval",
        "Interval 2"
    ]

    results = monte_carlo_compare(
        bot_strategies=bot_strategies,
        bot_names=bot_names,
        n_simulations=100000,
        seed=42,
    )

    print_results(results)
    plot_results(results)

print("\n" + "=" * 60)
print("Highest card advantage statistics")
print("=" * 60)

print(f"Times bot had unique highest card: {highest_card_opportunity}")
print(f"Times bot actually played it:      {highest_card_played}")

if highest_card_opportunity > 0:
    print(f"Usage rate: {highest_card_played / highest_card_opportunity:.4f}")