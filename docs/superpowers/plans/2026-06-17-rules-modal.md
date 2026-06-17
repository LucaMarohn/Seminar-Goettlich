# Rules Modal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bilingual DE/EN rules modal to the Hol's der Geier setup screen so first-time players can learn the game rules before starting.

**Architecture:** All changes are made inside `index.html`, which stores the entire app as a JSON-encoded HTML string inside `<script type="__bundler/template">`. Every task uses a Python script to decode that JSON, patch the HTML string, re-encode, and write back. No build tools exist — open the file directly in a browser to verify.

**Tech Stack:** Python 3 (standard library only), vanilla HTML/CSS/JS embedded in a single bundled HTML file.

---

## File Map

- Modify: `index.html` — the only file. Contains the full app as a JSON-encoded template string. All four tasks edit it sequentially.

---

## Task 1: Add Modal CSS

**Files:**
- Modify: `index.html` (inject `<style>` block before `</head>` in the template)

- [ ] **Step 1: Run the CSS injection script**

```bash
python3 << 'PYEOF'
import json

FILEPATH = 'index.html'

CSS = """
<style>
/* RULES MODAL */
#rules-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:1000;align-items:center;justify-content:center;padding:20px}
#rules-overlay.open{display:flex}
.rules-card{background:#1e1811;border:1px solid #6B5A43;border-radius:8px;width:100%;max-width:520px;max-height:85vh;display:flex;flex-direction:column;box-shadow:0 12px 48px rgba(0,0,0,.7);overflow:hidden}
.rules-header{display:flex;justify-content:space-between;align-items:center;padding:16px 20px;border-bottom:1px solid #3a3020;background:#251e17;flex-shrink:0}
.rules-title{font-family:'Cormorant Garamond',Georgia,serif;font-size:13px;letter-spacing:3px;color:#e8d9b8}
.rules-controls{display:flex;align-items:center;gap:12px}
.lang-pill{display:flex;background:#1a1410;border:1px solid #3a3020;border-radius:3px;overflow:hidden;font-size:11px;letter-spacing:1px;cursor:pointer;user-select:none}
.lang-pill span{padding:4px 10px;color:#8a7a65;transition:all .15s}
.lang-pill span.active{background:#c8a86a;color:#1a1410;font-weight:700}
.rules-close{color:#8a7a65;font-size:20px;cursor:pointer;line-height:1;padding:0 4px;background:none;border:none;font-family:inherit}
.rules-close:hover{color:#e8d9b8}
.rules-body{overflow-y:auto;padding:20px;flex:1}
.rules-section{margin-bottom:22px}
.rules-section-title{font-family:'Cormorant Garamond',Georgia,serif;font-size:10px;letter-spacing:3px;color:#c8a86a;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.rules-section-title::after{content:'';flex:1;height:1px;background:#3a3020}
.rules-text{font-family:'Cormorant Garamond',Georgia,serif;font-size:13px;line-height:1.7;color:#b8a888}
.rules-text b{color:#e8d9b8;font-weight:600}
.rules-example{background:#251e17;border:1px solid #3a3020;border-left:2px solid #c8a86a;border-radius:0 4px 4px 0;padding:10px 14px;margin-top:8px;font-family:'Cormorant Garamond',Georgia,serif;font-size:12px;color:#8a7a65;font-style:italic;line-height:1.6}
.rules-badge{display:inline-block;padding:1px 6px;border-radius:2px;font-size:11px;font-weight:700;font-family:monospace}
.rules-badge.pos{background:#1a3a1a;color:#6dbf6d;border:1px solid #2a5a2a}
.rules-badge.neg{background:#3a1a1a;color:#bf6d6d;border:1px solid #5a2a2a}
.rules-body[data-lang="en"] .de-text{display:none}
.rules-body[data-lang="en"] .en-text{display:block}
.rules-body .en-text{display:none}
</style>"""

with open(FILEPATH, 'r') as f:
    content = f.read()

MARKER = '<script type="__bundler/template">'
start_idx = content.find(MARKER) + len(MARKER)
end_idx = content.find('</script>', start_idx)
template = json.loads(content[start_idx:end_idx])

assert '</head>' in template, "Anchor not found"
template = template.replace('</head>', CSS + '\n</head>', 1)

new_json = json.dumps(template)
with open(FILEPATH, 'w') as f:
    f.write(content[:start_idx] + new_json + content[end_idx:])

print("CSS injected OK")
PYEOF
```

Expected output: `CSS injected OK`

- [ ] **Step 2: Verify**

```bash
python3 -c "
import json
with open('index.html') as f: c = f.read()
start = c.find('<script type=\"__bundler/template\">') + len('<script type=\"__bundler/template\">')
t = json.loads(c[start:c.find('</script>', start)])
print('rules-overlay found:', '#rules-overlay' in t)
print('lang-pill found:', '.lang-pill' in t)
"
```

Expected:
```
rules-overlay found: True
lang-pill found: True
```

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add rules modal CSS"
```

---

## Task 2: Add "?" Button Next to Start Button

**Files:**
- Modify: `index.html` (replace the start-button wrapper div in the template)

- [ ] **Step 1: Run the button replacement script**

```bash
python3 << 'PYEOF'
import json

FILEPATH = 'index.html'

OLD = '<div class="text-center mt-28">\n      <button class="btn btn-primary" onclick="startGame()" id="btn-start">✶ &nbsp;Karten austeilen&nbsp; ✶</button>\n    </div>'

NEW = '<div class="text-center mt-28" style="display:flex;justify-content:center;align-items:center;gap:10px;">\n      <button class="btn btn-primary" onclick="startGame()" id="btn-start">✶ &nbsp;Karten austeilen&nbsp; ✶</button>\n      <button class="btn btn-ghost" onclick="openRules()">?</button>\n    </div>'

with open(FILEPATH, 'r') as f:
    content = f.read()

MARKER = '<script type="__bundler/template">'
start_idx = content.find(MARKER) + len(MARKER)
end_idx = content.find('</script>', start_idx)
template = json.loads(content[start_idx:end_idx])

assert OLD in template, f"Anchor not found. Check whitespace."
template = template.replace(OLD, NEW, 1)

new_json = json.dumps(template)
with open(FILEPATH, 'w') as f:
    f.write(content[:start_idx] + new_json + content[end_idx:])

print("Button added OK")
PYEOF
```

Expected output: `Button added OK`

- [ ] **Step 2: Verify**

```bash
python3 -c "
import json
with open('index.html') as f: c = f.read()
start = c.find('<script type=\"__bundler/template\">') + len('<script type=\"__bundler/template\">')
t = json.loads(c[start:c.find('</script>', start)])
print('? button found:', 'onclick=\"openRules()\"' in t)
"
```

Expected: `? button found: True`

- [ ] **Step 3: Open `index.html` in a browser and confirm**

The setup screen should show `[ ✶ Karten austeilen ✶ ] [ ? ]` side by side. Clicking `?` will error (JS not added yet) — that is expected.

- [ ] **Step 4: Commit**

```bash
git add index.html
git commit -m "feat: add rules button to setup screen"
```

---

## Task 3: Add Modal HTML

**Files:**
- Modify: `index.html` (inject modal HTML before `.footer-note` div in the template)

- [ ] **Step 1: Run the modal HTML injection script**

```bash
python3 << 'PYEOF'
import json

FILEPATH = 'index.html'

MODAL_HTML = """
  <!-- RULES MODAL -->
  <div id="rules-overlay" onclick="closeRulesOnBackdrop(event)">
    <div class="rules-card" onclick="event.stopPropagation()">
      <div class="rules-header">
        <span class="rules-title" id="rules-title">DIE REGELN</span>
        <div class="rules-controls">
          <div class="lang-pill" onclick="toggleRulesLang()">
            <span id="rules-btn-de" class="active">DE</span>
            <span id="rules-btn-en">EN</span>
          </div>
          <button class="rules-close" onclick="closeRules()">&#215;</button>
        </div>
      </div>
      <div class="rules-body" id="rules-body" data-lang="de">

        <div class="rules-section">
          <div class="rules-section-title">I &middot; Ziel des Spiels</div>
          <div class="rules-text de-text">Wer am Ende die <b>meisten Punkte</b> gesammelt hat, gewinnt. Negative Punkte sind m&#246;glich &mdash; pass auf den Geier auf!</div>
          <div class="rules-text en-text">The player with the <b>most points</b> at the end wins. Negative scores are possible &mdash; beware the vulture!</div>
        </div>

        <div class="rules-section">
          <div class="rules-section-title">II &middot; Die Karten</div>
          <div class="rules-text de-text">Jeder Spieler erh&#228;lt ein <b>Handblatt von 1 bis 15</b> (15 Karten). In der Mitte liegen <b>15 Beutekarten</b>: <span class="rules-badge neg">&minus;5</span> bis <span class="rules-badge neg">&minus;1</span> und <span class="rules-badge pos">+1</span> bis <span class="rules-badge pos">+10</span> (keine Null), verdeckt gemischt.</div>
          <div class="rules-text en-text">Each player receives a <b>hand of cards 1 to 15</b> (15 cards). In the centre lie <b>15 prize cards</b>: <span class="rules-badge neg">&minus;5</span> to <span class="rules-badge neg">&minus;1</span> and <span class="rules-badge pos">+1</span> to <span class="rules-badge pos">+10</span> (no zero), shuffled face-down.</div>
        </div>

        <div class="rules-section">
          <div class="rules-section-title">III &middot; Jede Runde</div>
          <div class="rules-text de-text">Eine Beutekarte wird aufgedeckt &mdash; sie zeigt an, was auf dem Spiel steht. Alle Spieler w&#228;hlen <b>gleichzeitig und verdeckt</b> eine Karte aus ihrem Handblatt. Dann werden alle Gebote enth&#252;llt. Jede Handkarte kann nur <b>einmal</b> gespielt werden.</div>
          <div class="rules-text en-text">One prize card is turned face-up &mdash; it shows what is at stake. All players <b>simultaneously and secretly</b> choose one card from their hand. Then all bids are revealed. Each hand card may only be played <b>once</b>.</div>
        </div>

        <div class="rules-section">
          <div class="rules-section-title">IV &middot; <span class="rules-badge pos">+ Beutekarte</span> &mdash; Aufl&#246;sung</div>
          <div class="rules-text de-text">Die <b>h&#246;chste einzigartige</b> Handkarte gewinnt die Beute.<br><br><b>Wichtig:</b> Spielen mehrere Spieler die gleiche h&#246;chste Karte, heben sich diese Gebote gegenseitig auf und werden gestrichen. Dann z&#228;hlt die n&#228;chsth&#246;chste &mdash; bis eine einzigartige Karte &#252;brig bleibt.</div>
          <div class="rules-text en-text">The <b>highest unique</b> hand card wins the prize.<br><br><b>Note:</b> If multiple players play the same highest card, those bids cancel each other out and are removed. The next highest card is then checked &mdash; until one unique card remains.</div>
          <div class="rules-example de-text">Beispiel: Spieler A=9, B=9, C=7, D=5 &mdash; Die zwei Neunen heben sich auf. Spieler C gewinnt mit der 7.</div>
          <div class="rules-example en-text">Example: Player A=9, B=9, C=7, D=5 &mdash; The two nines cancel. Player C wins with the 7.</div>
        </div>

        <div class="rules-section">
          <div class="rules-section-title">V &middot; <span class="rules-badge neg">&minus; Beutekarte</span> &mdash; Aufl&#246;sung</div>
          <div class="rules-text de-text">Die <b>niedrigste einzigartige</b> Handkarte tr&#228;gt den Verlust.<br><br><b>Wichtig:</b> Gleiche niedrigste Karten heben sich auf, die n&#228;chstniedrigste z&#228;hlt weiter.</div>
          <div class="rules-text en-text">The <b>lowest unique</b> hand card takes the penalty.<br><br><b>Note:</b> Identical lowest cards cancel each other, and the next lowest is checked instead.</div>
          <div class="rules-example de-text">Beispiel: Spieler A=2, B=2, C=4, D=6 &mdash; Die zwei Zweien heben sich auf. Spieler C tr&#228;gt die Strafe mit der 4.</div>
          <div class="rules-example en-text">Example: Player A=2, B=2, C=4, D=6 &mdash; The two twos cancel. Player C takes the penalty with the 4.</div>
        </div>

        <div class="rules-section">
          <div class="rules-section-title">VI &middot; Kein Sieger &mdash; Weitergabe</div>
          <div class="rules-text de-text">Falls <b>alle Gebote gleichstehen</b> und kein einzigartiges Gebot &#252;brig bleibt, geht die Beutekarte an niemanden &mdash; sie <b>wandert in die n&#228;chste Runde</b> und summiert sich mit der n&#228;chsten Beutekarte. In der folgenden Runde ist der gesamte aufgelaufene Betrag zu gewinnen (oder zu verlieren).</div>
          <div class="rules-text en-text">If <b>all bids tie</b> and no unique bid remains, nobody wins the prize &mdash; it <b>carries over to the next round</b> and accumulates with the next prize card. In the following round, the total stacked amount is at stake.</div>
          <div class="rules-example de-text">Beispiel: Beutekarte +7, alle spielen die 10 &mdash; niemand gewinnt. N&#228;chste Beutekarte ist +3. Jetzt sind +10 im Topf.</div>
          <div class="rules-example en-text">Example: Prize card +7, everyone plays 10 &mdash; nobody wins. Next prize card is +3. Now +10 is at stake in the pot.</div>
        </div>

        <div class="rules-section">
          <div class="rules-section-title">VII &middot; Variante: Zwei Spieler</div>
          <div class="rules-text de-text">Bei nur zwei Spielern werden vor Spielbeginn <b>3 zuf&#228;llige Beutekarten</b> verdeckt aus dem Spiel entfernt &mdash; niemand wei&#223;, welche fehlen. Au&#223;erdem legt jeder Spieler <b>3 zuf&#228;llige Handkarten</b> verdeckt beiseite. Jeder startet mit 12 Handkarten, 12 Beutekarten sind im Spiel.</div>
          <div class="rules-text en-text">With only two players, <b>3 random prize cards</b> are removed face-down before the game &mdash; nobody knows which ones are missing. In addition, each player sets aside <b>3 random hand cards</b> face-down. Each player starts with 12 hand cards, and 12 prize cards are in play.</div>
        </div>

        <div class="rules-section">
          <div class="rules-section-title">VIII &middot; Spielende</div>
          <div class="rules-text de-text">Das Spiel endet, wenn alle Beutekarten gespielt wurden. Der Spieler mit dem <b>h&#246;chsten Punktestand</b> gewinnt. Bei Gleichstand gewinnen beide.</div>
          <div class="rules-text en-text">The game ends when all prize cards have been played. The player with the <b>highest score</b> wins. In case of a tie, both players share the victory.</div>
        </div>

      </div>
    </div>
  </div>

"""

ANCHOR = '<div class="footer-note">'

with open(FILEPATH, 'r') as f:
    content = f.read()

MARKER = '<script type="__bundler/template">'
start_idx = content.find(MARKER) + len(MARKER)
end_idx = content.find('</script>', start_idx)
template = json.loads(content[start_idx:end_idx])

assert ANCHOR in template, "Anchor not found"
template = template.replace(ANCHOR, MODAL_HTML + ANCHOR, 1)

new_json = json.dumps(template)
with open(FILEPATH, 'w') as f:
    f.write(content[:start_idx] + new_json + content[end_idx:])

print("Modal HTML injected OK")
PYEOF
```

Expected output: `Modal HTML injected OK`

- [ ] **Step 2: Verify**

```bash
python3 -c "
import json
with open('index.html') as f: c = f.read()
start = c.find('<script type=\"__bundler/template\">') + len('<script type=\"__bundler/template\">')
t = json.loads(c[start:c.find('</script>', start)])
print('rules-overlay div found:', 'id=\"rules-overlay\"' in t)
print('section VIII found:', 'VIII' in t)
print('two-player section found:', 'Zwei Spieler' in t)
"
```

Expected:
```
rules-overlay div found: True
section VIII found: True
two-player section found: True
```

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add rules modal HTML with bilingual content"
```

---

## Task 4: Add JS Functions

**Files:**
- Modify: `index.html` (inject JS before `syncSetupUI();` in the template script)

- [ ] **Step 1: Run the JS injection script**

```bash
python3 << 'PYEOF'
import json

FILEPATH = 'index.html'

JS = """
// ═════════════════════════════════════════════
//  RULES MODAL
// ═════════════════════════════════════════════
let rulesLang = 'de';

function openRules() {
  document.getElementById('rules-overlay').classList.add('open');
}

function closeRules() {
  document.getElementById('rules-overlay').classList.remove('open');
}

function closeRulesOnBackdrop(event) {
  if (event.target === document.getElementById('rules-overlay')) closeRules();
}

function toggleRulesLang() {
  rulesLang = rulesLang === 'de' ? 'en' : 'de';
  document.getElementById('rules-body').setAttribute('data-lang', rulesLang);
  document.getElementById('rules-btn-de').className = rulesLang === 'de' ? 'active' : '';
  document.getElementById('rules-btn-en').className = rulesLang === 'en' ? 'active' : '';
  document.getElementById('rules-title').textContent = rulesLang === 'de' ? 'DIE REGELN' : 'THE RULES';
}

"""

ANCHOR = 'syncSetupUI();\nrenderHistory();'

with open(FILEPATH, 'r') as f:
    content = f.read()

MARKER = '<script type="__bundler/template">'
start_idx = content.find(MARKER) + len(MARKER)
end_idx = content.find('</script>', start_idx)
template = json.loads(content[start_idx:end_idx])

assert ANCHOR in template, "Anchor not found"
template = template.replace(ANCHOR, JS + ANCHOR, 1)

new_json = json.dumps(template)
with open(FILEPATH, 'w') as f:
    f.write(content[:start_idx] + new_json + content[end_idx:])

print("JS injected OK")
PYEOF
```

Expected output: `JS injected OK`

- [ ] **Step 2: Verify**

```bash
python3 -c "
import json
with open('index.html') as f: c = f.read()
start = c.find('<script type=\"__bundler/template\">') + len('<script type=\"__bundler/template\">')
t = json.loads(c[start:c.find('</script>', start)])
print('openRules found:', 'function openRules()' in t)
print('closeRules found:', 'function closeRules()' in t)
print('toggleRulesLang found:', 'function toggleRulesLang()' in t)
print('closeRulesOnBackdrop found:', 'function closeRulesOnBackdrop' in t)
"
```

Expected:
```
openRules found: True
closeRules found: True
toggleRulesLang found: True
closeRulesOnBackdrop found: True
```

- [ ] **Step 3: Commit**

```bash
git add index.html
git commit -m "feat: add rules modal JS (open/close/language toggle)"
```

---

## Task 5: Manual Verification

**Files:** None — browser testing only.

- [ ] **Step 1: Open `index.html` in a browser**

Double-click the file or run:
```bash
open index.html
```

- [ ] **Step 2: Check the "?" button**

On the setup screen, confirm a `?` button appears to the right of `✶ Karten austeilen ✶`. Both should be vertically centred in a row.

- [ ] **Step 3: Open the modal**

Click `?`. A dark overlay should appear with the rules card centred on screen. Confirm:
- Header shows "DIE REGELN"
- `[DE][EN]` pill toggle visible with DE active (gold)
- `×` close button in top right
- All 8 sections visible when scrolling (I through VIII)
- Section titles have the gold colour and extending rule line

- [ ] **Step 4: Test language toggle**

Click `[EN]` in the pill toggle. Confirm:
- Header changes to "THE RULES"
- EN pill becomes gold, DE becomes faint
- All text switches to English
- Click `[DE]` — switches back to German

- [ ] **Step 5: Test close behaviour**

- Click `×` → modal closes
- Reopen, then click anywhere on the dim backdrop → modal closes
- Reopen, click inside the card → modal stays open

- [ ] **Step 6: Test language persistence**

- Toggle to EN, close, click `?` again → should reopen in EN (language persists in session)

- [ ] **Step 7: Final commit**

```bash
git add index.html
git commit -m "feat: complete rules modal — bilingual DE/EN with 8 sections"
```
