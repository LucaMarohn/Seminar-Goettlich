# Rules Modal — Design Spec
Date: 2026-06-17

## Overview

Add a bilingual (DE/EN) rules modal to the Hol's der Geier frontend so first-time players can understand the game before starting.

---

## Button

- Location: Setup screen (`#screen-setup`), next to the "✶ Karten austeilen ✶" button
- The two buttons sit in a centered row with a small gap
- Button label: `?`
- Style: `btn btn-ghost` (matching existing ghost button style)
- `onclick`: calls `openRules()`

```html
<div style="display:flex; justify-content:center; gap:10px; align-items:center;">
  <button class="btn btn-primary" onclick="startGame()" id="btn-start">✶ &nbsp;Karten austeilen&nbsp; ✶</button>
  <button class="btn btn-ghost" onclick="openRules()">?</button>
</div>
```

---

## Modal

### Behaviour
- Opens on `openRules()` call
- Closes by: clicking the × button, or clicking the dimmed backdrop
- Default language: DE
- Language persists within the session (toggling DE→EN stays EN if you close and reopen)

### Structure
```
Fixed overlay (position:fixed, inset:0, background:rgba(0,0,0,0.7), z-index:1000)
└── Centered card (max-width:520px, max-height:85vh, overflow-y:auto)
    ├── Header (sticky)
    │   ├── Title: "DIE REGELN" / "THE RULES"
    │   ├── [DE][EN] pill toggle
    │   └── × close button
    └── Scrollable body
        └── 8 rule sections (see below)
```

### Styling
- Matches game aesthetic: dark parchment background (`#1e1811`), Cormorant Garamond / Georgia serif font
- Section titles: small caps, `letter-spacing:3px`, gold (`#c8a86a`), with a horizontal rule extending right
- Body text: `#b8a888`, `font-size:13px`, `line-height:1.7`
- Examples: indented block with a left gold border, italic, faint text
- Badge chips for `+` (green-tinted) and `−` (red-tinted) value card references

---

## Language Toggle

- Pill toggle in modal header: `[DE][EN]`
- Active language has gold background, inactive is faint
- Toggling swaps:
  - Modal title text ("DIE REGELN" ↔ "THE RULES")
  - All `.de-text` / `.en-text` content blocks (CSS `display:none` swap)
- State held in a JS variable `let rulesLang = 'de'`

---

## Rules Content (both languages)

### I · Ziel des Spiels / Goal
- DE: Wer am Ende die meisten Punkte gesammelt hat, gewinnt. Negative Punkte sind möglich.
- EN: The player with the most points at the end wins. Negative scores are possible.

### II · Die Karten / The Cards
- DE: Jeder Spieler erhält Handkarten 1–15. Beutekarten: −5 bis −1 und +1 bis +10 (keine Null), verdeckt gemischt.
- EN: Each player receives hand cards 1–15. Prize cards: −5 to −1 and +1 to +10 (no zero), shuffled face-down.

### III · Jede Runde / Each Round
- DE: Beutekarte aufdecken → alle spielen gleichzeitig und verdeckt eine Handkarte → alle enthüllen. Jede Handkarte nur einmal spielbar.
- EN: Reveal prize card → all play a hand card simultaneously and secretly → all reveal. Each hand card can only be played once.

### IV · + Beutekarte — Auflösung / Positive Prize Card
- DE: Höchste einzigartige Handkarte gewinnt. **Wichtig:** Gleiche höchste Karten heben sich auf, nächsthöchste zählt weiter.
- EN: Highest unique hand card wins. **Note:** Identical highest cards cancel, next highest is checked instead.
- Example (DE): A=9, B=9, C=7, D=5 → Neunen heben sich auf → C gewinnt mit 7.
- Example (EN): A=9, B=9, C=7, D=5 → nines cancel → C wins with 7.

### V · − Beutekarte — Auflösung / Negative Prize Card
- DE: Niedrigste einzigartige Handkarte trägt den Verlust. Gleiches Aufhebungs-Prinzip gilt.
- EN: Lowest unique hand card takes the penalty. Same cancelling rule applies.
- Example (DE): A=2, B=2, C=4, D=6 → Zweien heben sich auf → C trägt die Strafe mit 4.
- Example (EN): A=2, B=2, C=4, D=6 → twos cancel → C takes penalty with 4.

### VI · Kein Sieger — Weitergabe / Carry-Over
- DE: Falls alle Gebote gleichstehen (kein einzigartiges Gebot), wandert die Beutekarte in die nächste Runde und summiert sich auf.
- EN: If all bids tie (no unique bid remains), the prize carries over and stacks with the next prize card.
- Example (DE): +7, alle spielen 10 → niemand gewinnt → nächste Karte +3 → jetzt +10 im Topf.
- Example (EN): +7, everyone plays 10 → nobody wins → next card +3 → now +10 at stake.

### VII · Variante: Zwei Spieler / Two-Player Variant
- DE: Vor Spielbeginn werden 3 zufällige Beutekarten verdeckt entfernt (niemand weiß welche). Jeder Spieler legt auch 3 zufällige Handkarten verdeckt beiseite. Jeder startet mit 12 Handkarten, 12 Beutekarten im Spiel.
- EN: Before the game, 3 random prize cards are removed face-down (nobody knows which). Each player also sets aside 3 random hand cards face-down. Each player starts with 12 hand cards, 12 prize cards in play.

### VIII · Spielende / End of Game
- DE: Nach allen Beutekarten: höchster Punktestand gewinnt. Bei Gleichstand gewinnen beide.
- EN: After all prize cards: highest score wins. Tied scores share the victory.

---

## Implementation Notes

### File structure
Single bundled HTML file (`index.html`). The actual source lives JSON-encoded inside `<script type="__bundler/template">`. All changes must be made by:
1. Python-decoding the JSON template string
2. Editing the HTML/CSS/JS within it
3. Re-encoding and replacing

### Changes required
1. **CSS**: Add modal overlay styles and rules card styles (appended to existing `<style>` block)
2. **HTML**: Add `<div id="rules-modal">` before closing `</div>` of `.sheet`, replace the start-button div with a flex row containing both buttons
3. **JS**: Add `let rulesLang = 'de'`, `openRules()`, `closeRules()`, `toggleRulesLang()` functions (appended to existing `<script>` block before `syncSetupUI()`)

### No external dependencies
All styling inline or via existing CSS variables. No new fonts or libraries needed.
