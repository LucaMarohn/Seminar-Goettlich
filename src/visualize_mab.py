"""
visualize_mab.py
================
Static infographic explaining the Multi-Armed Bandit assignment process
used in learning_intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Colour palette ────────────────────────────────────────────────────────────
C_LOW  = '#4878CF'   # blue
C_MID  = '#6ACC65'   # green
C_HIGH = '#D65F5F'   # red
C_CARD = '#F5F0E8'
C_DARK = '#2C2C2C'
C_GRAY = '#AAAAAA'

GROUP_COLORS = [C_LOW, C_MID, C_HIGH]
GROUP_NAMES  = ['low', 'mid', 'high']

SORTED_VALUE_CARDS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ── Simulate a plausible "after-learning" Q-table for illustration ────────────
rng = np.random.default_rng(7)
# Roughly: negative cards → high (sacrifice), mid-positive → low/mid, high-positive → high
true_best = [2, 2, 1, 1, 2,   # -5 … -1
             0, 0, 0, 1, 1,   #  1 …  5
             2, 2, 2, 1, 2]   #  6 … 10

Q = np.zeros((15, 3))
for i, g in enumerate(true_best):
    for k in range(3):
        base = 3.5 if k == g else 2.5
        Q[i, k] = base + rng.normal(0, 0.3)

counts = rng.integers(80, 300, size=(15, 3))

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 22), facecolor='white')
fig.suptitle(
    "Multi-Armed Bandit: How Each Value Card Learns Its Interval",
    fontsize=18, fontweight='bold', y=0.98, color=C_DARK
)

gs = fig.add_gridspec(
    4, 1,
    height_ratios=[2.2, 1.4, 1.2, 1.4],
    hspace=0.55,
    top=0.95, bottom=0.03, left=0.05, right=0.97
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  –  Bandit structure for 3 example value cards
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(0, 18)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title("① Each value card is an independent 3-armed bandit",
              fontsize=13, fontweight='bold', color=C_DARK, pad=8, loc='left')

example_vcs = [0, 5, 14]   # indices of -5, 1, 10
ex_labels   = ['-5', '1', '10']
ex_x        = [2, 9, 16]

for col, (vc_idx, label, cx) in enumerate(zip(example_vcs, ex_labels, ex_x)):
    # Value card box
    card = FancyBboxPatch((cx - 1.1, 7.5), 2.2, 1.8,
                          boxstyle="round,pad=0.15",
                          facecolor=C_CARD, edgecolor=C_DARK, linewidth=2)
    ax1.add_patch(card)
    ax1.text(cx, 8.4, f"Value card\n{label}",
             ha='center', va='center', fontsize=11, fontweight='bold', color=C_DARK)

    # 3 arm boxes
    arm_xs = [cx - 1.6, cx, cx + 1.6]
    for arm_i, (ax_x, gname, gc) in enumerate(zip(arm_xs, GROUP_NAMES, GROUP_COLORS)):
        q_val = Q[vc_idx, arm_i]
        n_val = counts[vc_idx, arm_i]
        is_best = (arm_i == true_best[vc_idx])

        arm_box = FancyBboxPatch((ax_x - 0.65, 4.2), 1.3, 2.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=gc, edgecolor='white' if is_best else gc,
                                 linewidth=3 if is_best else 1,
                                 alpha=0.85 if is_best else 0.45)
        ax1.add_patch(arm_box)

        ax1.text(ax_x, 6.55, f"Arm {arm_i}\n({gname})",
                 ha='center', va='center', fontsize=8.5,
                 fontweight='bold' if is_best else 'normal',
                 color='white')
        ax1.text(ax_x, 5.5, f"Q = {q_val:.2f}",
                 ha='center', va='center', fontsize=8, color='white')
        ax1.text(ax_x, 4.75, f"n = {n_val}",
                 ha='center', va='center', fontsize=7.5, color='white', alpha=0.9)

        if is_best:
            ax1.text(ax_x, 4.0, "▲ best", ha='center', va='center',
                     fontsize=8, fontweight='bold', color=gc)

        # Arrow from card to arm
        ax1.annotate('', xy=(ax_x, 7.05), xytext=(cx, 7.5),
                     arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.2))

    ax1.text(cx, 3.3,
             "ε-greedy: pull argmax Q\n(or random with prob ε)",
             ha='center', va='center', fontsize=8, color=C_DARK,
             style='italic',
             bbox=dict(facecolor='#EFEFEF', edgecolor=C_GRAY, boxstyle='round,pad=0.3'))

# Ellipsis between cards
ax1.text(5.5, 6.2, '· · · 13 more cards · · ·',
         ha='center', va='center', fontsize=10, color=C_GRAY, style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  –  One game: all 15 value cards choose a group
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(-0.5, 14.5)
ax2.set_ylim(-0.5, 3.5)
ax2.axis('off')
ax2.set_title("② At the start of every game: each card independently selects a group",
              fontsize=13, fontweight='bold', color=C_DARK, pad=8, loc='left')

chosen_groups = np.array(true_best)   # use our illustrative assignment

for i, (vc, g) in enumerate(zip(SORTED_VALUE_CARDS, chosen_groups)):
    color = GROUP_COLORS[g]
    card = FancyBboxPatch((i - 0.42, 1.2), 0.84, 1.6,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor='white', linewidth=1.5,
                          alpha=0.88)
    ax2.add_patch(card)
    ax2.text(i, 2.15, str(vc),
             ha='center', va='center', fontsize=9.5, fontweight='bold', color='white')
    ax2.text(i, 1.55, GROUP_NAMES[g][0].upper(),
             ha='center', va='center', fontsize=8, color='white', alpha=0.9)

# Bracket labels
n_low  = int(np.sum(chosen_groups == 0))
n_mid  = int(np.sum(chosen_groups == 1))
n_high = int(np.sum(chosen_groups == 2))

low_idxs  = [i for i, g in enumerate(chosen_groups) if g == 0]
mid_idxs  = [i for i, g in enumerate(chosen_groups) if g == 1]
high_idxs = [i for i, g in enumerate(chosen_groups) if g == 2]

for idxs, label, color in [
    (low_idxs,  f'Low group\n{n_low} cards',  C_LOW),
    (mid_idxs,  f'Mid group\n{n_mid} cards',  C_MID),
    (high_idxs, f'High group\n{n_high} cards', C_HIGH),
]:
    if not idxs:
        continue
    xmin, xmax = min(idxs) - 0.45, max(idxs) + 0.45
    ax2.plot([xmin, xmin, xmax, xmax], [1.05, 0.9, 0.9, 1.05],
             color=color, lw=1.8)
    ax2.text((xmin + xmax) / 2, 0.6, label,
             ha='center', va='center', fontsize=8, color=color, fontweight='bold')

ax2.text(7, 3.25,
         "Cards of the same color → same group  (order within a group doesn't matter)",
         ha='center', va='center', fontsize=9, color=C_DARK, style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  –  Interval derivation
# ─────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.set_xlim(0, 18)
ax3.set_ylim(0, 5)
ax3.axis('off')
ax3.set_title("③ Compute hand-card intervals from group sizes",
              fontsize=13, fontweight='bold', color=C_DARK, pad=8, loc='left')

# Formula boxes
formula_data = [
    (3,   C_LOW,  f'Low  ({n_low} cards)',  f'hand [1, {n_low}]'),
    (9,   C_MID,  f'Mid  ({n_mid} cards)',  f'hand [{n_low+1}, {n_low+n_mid}]'),
    (15,  C_HIGH, f'High ({n_high} cards)', f'hand [{n_low+n_mid+1}, 15]'),
]

for cx, color, top_label, bot_label in formula_data:
    box = FancyBboxPatch((cx - 2.5, 0.8), 5, 3.2,
                         boxstyle="round,pad=0.2",
                         facecolor=color, edgecolor='white', linewidth=2, alpha=0.2)
    ax3.add_patch(box)
    ax3.text(cx, 3.3, top_label,
             ha='center', va='center', fontsize=11, color=color, fontweight='bold')
    ax3.text(cx, 2.1, '↓', ha='center', va='center', fontsize=16, color=color)
    ax3.text(cx, 1.2, bot_label,
             ha='center', va='center', fontsize=12, color=color, fontweight='bold',
             bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3', lw=2))

ax3.text(9, 4.7,
         f"Group sizes: Low={n_low}, Mid={n_mid}, High={n_high}  →  "
         f"Intervals always cover all 15 hand cards, no gaps",
         ha='center', va='center', fontsize=9.5, color=C_DARK, style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  –  Play a round + Q-value update
# ─────────────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3])
ax4.set_xlim(0, 18)
ax4.set_ylim(0, 6)
ax4.axis('off')
ax4.set_title("④ During the game: use the interval — then update Q-values from the game reward",
              fontsize=13, fontweight='bold', color=C_DARK, pad=8, loc='left')

# Example: value card 10 appears → high group → play from [n_low+n_mid+1, 15]
# Show the chain: value card → lookup group → lookup interval → pick hand card

steps = [
    (1.5,  '#F5F0E8', C_DARK, "Value card\n10 revealed",         ''),
    (5.5,  C_HIGH,    'white', f"Group: high\n(learned)",         '→'),
    (9.5,  C_HIGH,    'white', f"Interval:\n[{n_low+n_mid+1}, 15]", '→'),
    (13.5, '#E8F4E8', C_DARK,  "Play random\ncard from\ninterval", '→'),
]

for cx, fc, tc, label, arrow in steps:
    if arrow:
        ax4.annotate('', xy=(cx - 1.6, 3.5), xytext=(cx - 2.8, 3.5),
                     arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2))
    box = FancyBboxPatch((cx - 1.5, 2.0), 3.0, 3.0,
                         boxstyle="round,pad=0.2",
                         facecolor=fc, edgecolor=C_DARK, linewidth=1.5)
    ax4.add_patch(box)
    ax4.text(cx, 3.5, label,
             ha='center', va='center', fontsize=10, color=tc, fontweight='bold')

# Q-update formula
ax4.annotate('', xy=(16.8, 3.5), xytext=(15.2, 3.5),
             arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2))
ax4.text(17.2, 4.8, "After game ends:",
         ha='center', va='center', fontsize=9, color=C_DARK, style='italic')
update_box = FancyBboxPatch((15.3, 1.5), 2.5, 2.7,
                             boxstyle="round,pad=0.2",
                             facecolor='#FFF8E7', edgecolor='#E0A800', linewidth=2)
ax4.add_patch(update_box)
ax4.text(16.55, 3.6, "Q-update",
         ha='center', va='center', fontsize=9, fontweight='bold', color='#A07000')
ax4.text(16.55, 2.8, "Q[vc, arm] +=",
         ha='center', va='center', fontsize=8, color=C_DARK)
ax4.text(16.55, 2.15, "(R − Q) / n",
         ha='center', va='center', fontsize=9, fontweight='bold', color='#A07000')
ax4.text(16.55, 1.65, "incremental mean",
         ha='center', va='center', fontsize=7.5, color=C_GRAY, style='italic')

# Loop arrow at bottom
ax4.annotate('', xy=(0.8, 0.7), xytext=(17.2, 0.7),
             arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.8,
                             connectionstyle='arc3,rad=-0.3'))
ax4.text(9, 0.25, "Repeat for thousands of games — Q-values converge, each value card commits to its best group",
         ha='center', va='center', fontsize=9, color=C_GRAY, style='italic')

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=C_LOW,  label='Low  group'),
    mpatches.Patch(facecolor=C_MID,  label='Mid  group'),
    mpatches.Patch(facecolor=C_HIGH, label='High group'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3,
           fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.005))

plt.savefig('/Users/lucamarohn/Seminar-Goettlich/mab_process.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("Saved to mab_process.png")
