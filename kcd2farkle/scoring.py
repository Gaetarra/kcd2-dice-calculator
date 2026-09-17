"""KCD2 Farkle scoring: what can you hold from a roll, and what is it worth?

Rules (https://kingdom-come-deliverance.fandom.com/wiki/Dice/KCD2):

    each 1                    100
    each 5                     50
    three 1s..three 6s        1000 200 300 400 500 600
    each die past the third   doubles that triple (x2, x4, x8)
    straight 1-5               500
    straight 2-6               750
    straight 1-6              1500

No three-pairs and no two-triplets rule -- KCD2 does not have them.

A hold must use *every* die selected, and is worth its best decomposition, so
four 1s is 2000 (quad) rather than 1100 (triple + single).

Counts throughout are 7-vectors over (joker, 1, 2, 3, 4, 5, 6).
"""

from functools import lru_cache
from itertools import product

JOKER = 0
TRIPLE = (0, 1000, 200, 300, 400, 500, 600)  # indexed by face 1..6
STRAIGHTS = ((1, 6, 1500), (2, 6, 750), (1, 5, 500))
SINGLES = {1: 100, 5: 50}

# Can a joker stand alone as a 1 or a 5?
#
# No: the wiki states a joker "matches any combination but never scores on its
# own".  What stays inferred is the edge cases that sentence does not cover --
# notably whether jokers alone can form a combination between themselves, which
# this allows.
#
# Flip to True to score lone jokers, then delete _tables6.npz and _tables7.npz so
# the cached roll tables get rebuilt.
JOKER_SCORES_ALONE = False


@lru_cache(maxsize=None)
def score_hold(held):
    """Best score for holding exactly `held`, or None if it cannot all be used.

    Jokers substitute for any face inside an n-of-a-kind or a straight.  They
    may not stand in for a lone 1 or a lone 5: the wiki says a joker "matches
    any combination but never scores on its own".
    ponytail: joker semantics inferred from the wiki's three worked examples,
    not from game code.  If the game disagrees, it is wrong only in here.
    """
    if not any(held):
        return 0
    jokers = held[JOKER]
    best = None

    def better(points, rest):
        nonlocal best
        sub = score_hold(rest)
        if sub is not None and (best is None or points + sub > best):
            best = points + sub

    for lo, hi, points in STRAIGHTS:
        missing = [f for f in range(lo, hi + 1) if held[f] == 0]
        if len(missing) > jokers:
            continue
        rest = list(held)
        rest[JOKER] -= len(missing)
        for f in range(lo, hi + 1):
            if rest[f]:
                rest[f] -= 1
        better(points, tuple(rest))

    for face in range(1, 7):
        for total in range(3, held[face] + jokers + 1):
            lo_j = max(0, total - held[face])
            for used_j in range(lo_j, min(jokers, total) + 1):
                rest = list(held)
                rest[JOKER] -= used_j
                rest[face] -= total - used_j
                better(TRIPLE[face] * (1 << (total - 3)), tuple(rest))

    for face, points in SINGLES.items():
        if held[face]:
            rest = list(held)
            rest[face] -= 1
            better(points, tuple(rest))
        elif JOKER_SCORES_ALONE and jokers:
            rest = list(held)
            rest[JOKER] -= 1
            better(points, tuple(rest))

    return best


def _dominates(a, b, counts):
    """Does option `a` = (points, held) weakly dominate `b`?

    Holding fewer dice for at least as many points is better, because whatever
    you leave on the table gets rerolled.  Clearing the table entirely is the
    exception: it is hot dice, so you reroll all six, which beats every other
    outcome at equal points.
    """
    (pa, ha), (pb, hb) = a, b
    if pa < pb:
        return False
    hot_a = ha == counts
    hot_b = hb == counts
    if hot_b and not hot_a:
        return False
    if hot_a and not hot_b:
        return True
    return all(x <= y for x, y in zip(ha, hb))


@lru_cache(maxsize=None)
def hold_options(counts):
    """Pareto-optimal ways to hold from a roll, as ((points, held), ...).

    Empty tuple means a bust.  Sorted by points descending.
    """
    best = {}
    for held in product(*(range(c + 1) for c in counts)):
        if not any(held):
            continue
        points = score_hold(held)
        if points is not None and points > best.get(held, -1):
            best[held] = points
    options = [(p, h) for h, p in best.items()]
    keep = [
        a for i, a in enumerate(options)
        if not any(j != i and _dominates(b, a, counts) and (b[0], b[1]) != (a[0], a[1])
                   for j, b in enumerate(options))
    ]
    # ties survive the filter above as duplicates only if identical; dedupe
    keep = sorted(set(keep), key=lambda o: (-o[0], o[1]))
    return tuple(keep)


def is_bust(counts):
    return not hold_options(counts)
