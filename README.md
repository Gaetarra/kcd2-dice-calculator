# KCD2 Dice Calculator

An exact solver for the dice mini-game (Farkle) in *Kingdom Come: Deliverance II*.

**Double-click `KCD2-Farkle.bat`**, or just open `KCD2-Dice-Calculator.html` in a browser. One HTML
file, no dependencies, no build step.

The batch file only exists to serve the page over `http://127.0.0.1`: Chrome and Edge refuse to start
Web Workers on a `file://` page, and without workers the search runs on one core instead of all of
them. Nothing leaves the machine. The page detects which mode it got and says so on the progress line.

## What it does

**Best dice combination** — tell it which dice you own and how many of each, and it finds the six that
score the most per turn. By default every multiset your inventory allows is tried; nothing is sampled
or truncated. You always have Ordinary dice, so if you own fewer than six specials the rest are filled
with plain ones and any inventory is searchable.

Sets are handed out best-first, so the table is correct for everything searched so far and **Stop** is
always a sane thing to press. A line under the progress bar says how many sets your inventory implies
and roughly how long they take, before you commit to anything.

Own most of the dice and that is hours. Tick **Neural Network Accelerator** and it is seconds — see
below.

**Score or pass** — enter the six dice on your table and the faces you just threw, with your score and
the points banked this turn. It tells you which dice to keep and whether to throw again or bank, with
the value and bust chance of every alternative. A *play* button on each line applies that move to the
board so you can walk a whole turn through it.

**Dice table** — the 43 dice with exact face weights and how strong each is on its own.

## How it works

Every value this reports is computed exactly. There is no simulation and no sampling anywhere; the
optional accelerator decides *which* sets to look at and never what one is worth.

- Every possible throw is enumerated with its exact probability.
- Every legal keep is scored by finding the best partition of the kept dice into combinations.
- Score-or-pass is solved by backward induction over *(dice remaining, points banked)*, including
  throws of fewer than six dice and hot-dice restarts, to the end of the turn.

Because every legal keep scores at least 50, the recursion only ever refers to a strictly higher
banked total. Sweeping the points grid downwards therefore makes every lookup land on a value that is
already final — one exact pass, no value iteration, no convergence tolerance.

### The match target is the whole endgame

The turn value is capped at the match target, because points past it do not help you win. That is what
actually ends a turn. Needing 100 and throwing a single 1, it banks and wins rather than chasing EV;
holding 1100 of the 1500 you need with one die left, it banks, because throwing that die is worth 459.
Set the target to ~20000 for a plain uncapped points-per-turn ranking.

### Speed

A six-distinct-type set solves in about 45 ms, so with workers on a 20-thread machine:

| you own | sets | search |
|---|---|---|
| 8 specials, one each | 247 | instant |
| 12 specials, one each | 2,510 | ~10 s |
| 20 specials, one each | 60,460 | ~4 min |
| all 32 distinct, six apiece | 2,324,784 | ~1.5 h, or press Stop |

The accelerator collapses the right-hand column to seconds at any size.

Two things pay for that: which keeps a throw allows depends only on the *shape* of the throw, not on
which die produced each face, so that table is built once and shared by every set; and the value grid
is padded by the largest possible keep so the inner loop is a single array read with no bounds test.

Eleven of the 43 dice are re-skins — six plain dice, Ci/Fer/Lu, three Painter's dice that match the
Unlucky die, and Balatro's die which rolls exactly like a Devil's head. Identical weights play
identically, so each group collapses to one representative: 43 dice, 32 that actually differ.

### Search order

Sets are ordered by the sum of their dice's strength measured as *three of it beside three plain
dice*, not as six of it alone. A die is nearly always used in a mix and some of the best ones are poor
six-at-a-time — the Pie die cannot roll a 5 or a 6 and is dreadful on its own, yet it is in the best
set in the game. On a 247-set inventory that ordering brings the true top ten inside the first 15
tried; ordering by solo strength takes 24. It only affects the order, never the answer: the search is
exhaustive either way.

## Neural Network Accelerator

A six-die set is a **multiset**, so the natural architecture is NNUE's: one embedding row per die
type, summed, then a small head.

```
accum = b0 + sum of W0[die] over the six dice      43 x 256
value = w2 . relu(W1 . relu(accum) + b1) + b2      256 x 128 -> 1
```

Summing the rows is what makes it blind to the order of your dice, which is right, because a set has
none — it cannot even represent an ordering. 44,289 parameters, trained offline against the exact
solver, so its targets were ground truth.

It is trained on **every loadout that exists**. The 43 dice collapse to 32 distinct weight vectors,
and a loadout is a six-multiset of those, so the whole domain is C(37,6) = **2,324,784** sets — not a
sample of something infinite, a finite list. Labelling all of it took 37 minutes on 20 threads. That
changes what the error figures mean: there is no held-out set, because there are no unseen sets, so
the numbers below are measured on the entire domain rather than estimated from a sample.

| | typical error | RMSE | worst under-prediction |
|---|---|---|---|
| 13,955 params, 200,000 sets | 2.45 | 9.03 | **2847.05** |
| 44,289 params, all 2,324,784 | **0.89** | **1.24** | **40.60** |

The old model's worst case is the important number there. It shipped with a 71 EV margin over a
2,847 EV hole, and nothing had ever measured it, because its holdout was drawn uniformly and a
uniform draw almost never contains a strong loadout. The new figure of 40.60 is a **bound**: no
loadout anywhere is under-rated by more than that, and there is no sampling left to be lucky about.

Capacity only helped once the data was there. On 120,000 sets, going 128x64 -> 256x128 made the worst
case *worse* (77 -> 145 EV); on the full domain it halves it (82 -> 41). Data first, then size.

The weights are baked into the page as 231 KB of base64; nothing trains in your browser, and the
workers each get a copy so the scoring is parallel too.

A learned filter cannot prove it did not throw away the winner, so it is never trusted on its own:

1. It scores every set in your search — 2.3 million of them in about a second across the workers.
2. The search then solves **a sample of your own sets exactly** and fits the network's output to what
   they were really worth. This step is needed because the network predicts uncapped points per turn
   while the page plays to a match target, which bends the top of the range down.
3. Sets are then opened **exactly**, in fitted order, and the sweep stops only when the next fitted
   value plus the margin is still below the worst set on the board — at which point nothing left can
   get on it. Because the sweep runs in descending fitted order, that stopping rank can be found by
   bisection, which is what the progress bar counts towards.
4. Every set it opens is a fresh test. A miss larger than the margin widens the margin on the spot,
   which is safe because nothing has been skipped yet — the sweep only ever skips its own tail, and
   only after this check has passed for everything above it.

### The margin

`margin = max(3 x worst miss in the sample, 40.60, 25)`

Two different errors have to be covered. The network's own is **proven** — 40.60 EV over every
loadout that exists — and capping only compresses (d capped / d uncapped <= 1), so that bound carries
through the fit. The fit onto your cap is the part that is merely sampled, and three times the worst
the sample showed covers it. Taking the larger means a lucky sample cannot talk the margin below what
the network is known to be capable of.

### Fitting the cap

Three things about that fit are not obvious, and each one was a bug first:

- **The sample is stratified**, half from the top by prediction and half uniform. A uniform sample
  measures typical error honestly, but the fit built from it gets *used* on the strongest loadouts,
  and a uniform draw of a few hundred from 2.3 million contains none of them. Fitted over predictions
  374-3177 and then evaluated at 10,422, the parabola had long since turned over: the true best
  loadout came out ranked **last of 2,324,784** and the sweep opened the space from the wrong end.
- **The regressor is bounded**: `p*cap/(p+cap)`, not `p`. Raw predictions reach 10,426 while capped
  value cannot pass the cap, so raw `p` gives the few extreme sets least-squares leverage of order
  `p^2` and bends the curve through the middle, where nearly every set lives. This transform stays
  monotone in `p` forever, so it still tells the leaders apart — `min(p, cap)` ties them and loses
  2.4x more pruning — while being bounded, which removes the leverage. It is standardised before the
  fit, because the 3x3 normal equations lose their precision otherwise.
- **The fit is never evaluated past its vertex.** Capped value rises with uncapped, so a fit that
  falls is wrong by construction; clamping keeps it flat and groups those sets at the top.

Measured end to end, single-threaded so the comparison is clean:

| inventory | sets | opened | margin | result |
|---|---|---|---|---|
| 2,510 sets | 2,510 | **30 (1.20%)** | 41 | same top twelve as exhaustive, 8.1 s against 174 s |
| everything, six apiece | 2,324,784 | **660 (0.028%)** | 232 | 7.6 s against ~446 min, margin never widened |

The ⓘ button reports all of that after every run.

That is a statistical guarantee, not a proof — the network's own error is bounded, but the fit onto
your cap is sampled. Untick the box for the proof; exact stays the default.

## Dice data

The wiki lists face chances rounded to one decimal; the game uses small integer weights. All 43 were
recovered exactly by rational reconstruction — the smallest denominator reproducing every published
percentage to within 0.1 — worst residual 0.073 percentage points. Denominators land between 6 and 32.

```
Ordinary die       [1,1,1,1,1,1]/6
Weighted die       [10,1,1,1,1,1]/15     66.7% ones
Aranka's die       [6,1,6,1,6,1]/21
King's die         [4,6,7,8,4,3]/32
Pie die            [6,1,3,3,0,0]/13      cannot roll a 5 or a 6
```

**The two joker dice.** The Devil's head die has a devil's head where its 1 should be. Balatro's die
has Jimbo's grin in the same place — the wiki shows a joker on all six faces, which is an unfilled
template, not the game; the two dice are identical. A joker substitutes into any combination but never
scores on its own, so it can complete a triple, a quad or a straight, but is never a lone 1 or 5, and
never forms a combination purely among other jokers.

## Correctness

A self-check runs on page load; its result is at the bottom of the Dice table tab. It covers the
scoring table, the dice data, and the two turn numbers below.

The chance of busting on six plain dice comes out as 1440/46656 = 3.0864%, which matches the hand
derivation. Six plain dice average **564.4376** points per turn — and the separate Python engine in
`kcd2farkle/`, which shares no code with this page, independently computes 564.437620. Two solvers in
two languages agreeing to four decimals is the real check on the DP.

`run-tests.bat` runs that Python suite: it re-derives every die's weights from the published
percentages, brute-forces the roll tables, and plays 40,000 simulated turns against the computed
value. It needs the `.venv` it builds on first run (numpy + numba) and takes a couple of minutes. The
HTML page needs none of it.

## Known limits

- It maximises your own chance of getting to the target and does not model the opponent's score. When
  they are one turn from winning you should gamble harder than this advises.
- The accelerator's margin is only partly proven. The network's own error is bounded over the whole
  domain, but the fit onto your match target is re-derived from your own sets each run, so a set far
  down the predicted order could still in principle beat its fitted value by more than the margin.
  Exact mode is the proof and is the default.
- Capped value is not a function of uncapped value, so no fit of the network's output can be exact:
  two loadouts with the same uncapped EV cap differently depending on how reliably they reach the
  target. Within a narrow band of prediction, true capped value still spreads 35-120 EV, and that is
  the floor on the sampled half of the margin. Training the network on the cap directly would remove
  it.
- Joker behaviour is inferred from the wiki's examples, not from game code. The two engines differ on
  one unreachable case: three or more jokers with no real die. This page calls that no score; the
  Python engine scores it. You cannot own three joker dice, so nothing you can actually play differs.
- Badges are not modelled. They would change the set of legal moves, not the shape of the solver.

## Data source

Scoring rules, dice list and face chances from the
[Kingdom Come: Deliverance wiki](https://kingdom-come-deliverance.fandom.com/wiki/Dice/KCD2).
