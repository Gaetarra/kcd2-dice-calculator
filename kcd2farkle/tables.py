"""Precomputed roll -> hold-option tables, shared by every dice combination.

The key observation that makes the exhaustive search affordable: which holds a
roll offers depends on *where the faces landed*, not on what kind of dice they
are.  Die types only set the probabilities.  So the whole outcome -> options
structure is built once, globally, and every combination reuses it.

On top of that, outcomes collapse hard.  Two rolls that differ only in their
non-scoring dice offer exactly the same holds, so they share an entry.  Grouping
outcomes by their option list gives a measured ~17x reduction (46,656 six-dice
rolls -> 2,781 distinct option lists), which is most of the reason a 4M-combo
exhaustive search finishes in minutes rather than days.

Layout: positions are the six dice slots, a mask is which of them are still on
the table, and an outcome index is base-7 digits over the masked positions
(digit 0 = joker, 1..6 = the face).  Scores are stored divided by 50.
"""

import functools
import itertools
import os
from collections import defaultdict

import numpy as np

from scoring import hold_options

N_POS = 6
FULL = (1 << N_POS) - 1
_HERE = os.path.dirname(os.path.abspath(__file__))

# Two tables.  Most combinations hold no joker die, and for those the six-face
# table is half the size and half the work, so it is the default path.
PLAIN, JOKER = 6, 7
CACHE = {PLAIN: os.path.join(_HERE, "_tables6.npz"), JOKER: os.path.join(_HERE, "_tables7.npz")}


@functools.lru_cache(maxsize=None)
def _options_for(positions, faces):
    """All Pareto-optimal (points//50, remaining_mask) holds for one outcome."""
    counts = [0] * 7
    for f in faces:
        counts[f] += 1
    full_mask = 0
    for p in positions:
        full_mask |= 1 << p

    by_face = defaultdict(list)
    for p, f in zip(positions, faces):
        by_face[f].append(p)

    best = {}
    for points, held in hold_options(tuple(counts)):
        picks = [itertools.combinations(by_face[f], held[f]) for f in range(7) if held[f]]
        for choice in itertools.product(*picks):
            rem = full_mask
            for group in choice:
                for p in group:
                    rem &= ~(1 << p)
            if best.get(rem, -1) < points:
                best[rem] = points

    items = list(best.items())

    def dominated(rem, pts):
        for rem2, pts2 in items:
            if rem2 == rem:
                continue
            if pts2 < pts:
                continue
            # rem2 == 0 is hot dice: reroll all six, better than any leftover set
            if rem2 == 0 or (rem != 0 and rem2 & rem == rem):
                return True
        return False

    keep = [(p // 50, r) for r, p in items if not dominated(r, p)]
    return tuple(sorted(keep))


def build(n_face=PLAIN):
    """Build every mask's outcome -> class -> options table.

    n_face=6 covers ordinary dice; n_face=7 adds the joker face at digit 0.
    """
    face_of = (lambda d: d + 1) if n_face == PLAIN else (lambda d: d)
    cls_of, cls_opt_off, opt_pts, opt_rem = [], [0], [], []
    mask_out_off, mask_cls_off = [0], [0]
    mask_pos = np.zeros((1 << N_POS, N_POS), np.int8)
    mask_k = np.zeros(1 << N_POS, np.int8)

    for mask in range(1 << N_POS):
        positions = [p for p in range(N_POS) if mask >> p & 1]
        k = len(positions)
        mask_k[mask] = k
        for i, p in enumerate(positions):
            mask_pos[mask, i] = p

        seen = {}
        for outcome in range(n_face ** k):
            faces = [face_of((outcome // n_face ** i) % n_face) for i in range(k)]
            key = _options_for(positions, faces)
            if key not in seen:
                seen[key] = len(seen)
                for points, rem in key:
                    opt_pts.append(points)
                    opt_rem.append(rem)
                cls_opt_off.append(len(opt_pts))
            cls_of.append(seen[key])
        mask_out_off.append(len(cls_of))
        mask_cls_off.append(len(cls_opt_off) - 1)

    return dict(
        cls_of=np.array(cls_of, np.int32),
        cls_opt_off=np.array(cls_opt_off, np.int32),
        opt_pts=np.array(opt_pts, np.int16),
        opt_rem=np.array(opt_rem, np.int8),
        mask_out_off=np.array(mask_out_off, np.int64),
        mask_cls_off=np.array(mask_cls_off, np.int32),
        mask_pos=mask_pos,
        mask_k=mask_k,
    )


def load(n_face=PLAIN):
    path = CACHE[n_face]
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    t = build(n_face)
    np.savez_compressed(path, **t)
    return t


if __name__ == "__main__":
    import time

    for n_face in (PLAIN, JOKER):
        t0 = time.time()
        t = build(n_face)
        np.savez_compressed(CACHE[n_face], **t)
        n_cls = len(t["cls_opt_off"]) - 1
        print(f"{n_face}-face table, built in {time.time() - t0:.1f}s")
        print(f"  outcomes     {len(t['cls_of']):,}")
        print(f"  classes      {n_cls:,}  ({len(t['cls_of']) / n_cls:.1f}x reduction)")
        print(f"  option slots {len(t['opt_pts']):,}   <- this x n_acc is the DP cost")
        print(f"  cache        {os.path.getsize(CACHE[n_face]) / 1e6:.2f} MB")
