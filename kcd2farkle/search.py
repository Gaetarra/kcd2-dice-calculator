"""Exhaustive search for the best six-die loadout.

Every 6-multiset of the dice you own gets its exact turn EV from engine.py --
no heuristics, no sampling, nothing skipped.  Work is split across processes;
each worker keeps only its own top few and ships those back, so the parent never
holds millions of results.
"""

import heapq
import itertools
import multiprocessing as mp
import time

import numpy as np

import engine
from dice_data import NAMES

CHUNK = 400  # combos per task: big enough to amortise IPC, small enough for smooth progress


def _types(inventory):
    """[(die index, how many you own)] for everything with a positive count."""
    return [(NAMES.index(n), q) for n, q in sorted(inventory.items()) if q > 0]


def count_combos(inventory):
    """How many distinct six-die loadouts the inventory allows."""
    poly = [1]
    for _, qty in _types(inventory):
        nxt = [0] * (len(poly) + min(qty, 6))
        for i, c in enumerate(poly):
            for take in range(min(qty, 6) - 0 + 1):
                if i + take < len(nxt):
                    nxt[i + take] += c
        poly = nxt[:7]
    return poly[6] if len(poly) > 6 else 0


def enumerate_combos(inventory):
    """Every allowed six-die loadout, as a sorted tuple of die indices."""
    types = _types(inventory)

    def walk(i, left, acc):
        if left == 0:
            yield tuple(acc)
            return
        if i == len(types):
            return
        idx, qty = types[i]
        for take in range(min(qty, left), -1, -1):
            yield from walk(i + 1, left - take, acc + [idx] * take)

    return walk(0, 6, [])


def chunks(it, size=CHUNK):
    it = iter(it)
    while True:
        block = list(itertools.islice(it, size))
        if not block:
            return
        yield np.array(block, np.int8)


_ENGINE = None
_KEEP = 8  # top combos each worker reports per chunk


def _init(a_max):
    global _ENGINE
    _ENGINE = engine.Engine(a_max=a_max)


def _work(block):
    ev = [(_ENGINE.turn_ev([NAMES[i] for i in c]), tuple(int(i) for i in c)) for c in block]
    return len(block), heapq.nlargest(_KEEP, ev)


def _work_all(block):
    return np.array([_ENGINE.turn_ev([NAMES[i] for i in c]) for c in block])


def evaluate(combos, a_max=engine.DEFAULT_A_MAX, processes=None, progress=None):
    """Exact turn EV for each of `combos` (an (n, 6) array of die indices).

    Keeps every value, unlike search(), which is what label generation needs.
    """
    combos = np.asarray(combos, np.int8)
    processes = processes or mp.cpu_count()
    out, done, t0 = [], 0, time.time()
    with mp.Pool(processes, initializer=_init, initargs=(a_max,)) as pool:
        for part in pool.imap(_work_all, chunks(iter(combos))):
            out.append(part)
            done += len(part)
            if progress:
                progress(done, len(combos), time.time() - t0)
    return np.concatenate(out) if out else np.zeros(0)


def search(inventory, top=25, a_max=engine.DEFAULT_A_MAX, processes=None, progress=None,
           should_stop=None):
    """Rank every allowed loadout by exact turn EV.  Returns the best `top`.

    `progress(done, total, elapsed)` is called as chunks land; `should_stop()`
    lets a UI cancel.  Results are (turn_ev, [die names]).
    """
    total = count_combos(inventory)
    if total == 0:
        return []
    processes = processes or mp.cpu_count()
    best, done, t0 = [], 0, time.time()

    with mp.Pool(processes, initializer=_init, initargs=(a_max,)) as pool:
        results = pool.imap_unordered(_work, chunks(enumerate_combos(inventory)))
        for n, part in results:
            done += n
            for item in part:
                if len(best) < top:
                    heapq.heappush(best, item)
                elif item > best[0]:
                    heapq.heapreplace(best, item)
            if progress:
                progress(done, total, time.time() - t0)
            if should_stop and should_stop():
                pool.terminate()
                break

    return [(ev, [NAMES[i] for i in c]) for ev, c in sorted(best, reverse=True)]


def all_combos(inventory):
    """Every allowed loadout as one (n, 6) int8 array."""
    arr = np.empty((count_combos(inventory), 6), np.int8)
    for i, c in enumerate(enumerate_combos(inventory)):
        arr[i] = c
    return arr


def search_fast(inventory, top=25, a_max=engine.DEFAULT_A_MAX, processes=None,
                progress=None, should_stop=None, block=None, model=None):
    """Surrogate-guided search: rank by prediction, then verify from the top down.

    Not "score everything and trust the top N".  Loadouts are exactly evaluated
    in predicted order and the sweep stops only once the *next* prediction plus
    the model's calibrated margin is still below the best EV actually found --
    at which point nothing left can beat it, unless the margin itself is wrong.
    So this is exact given the calibration holds, and it reports how much of the
    space it had to open to get there.
    """
    import surrogate

    model = model or surrogate.Surrogate.load()
    if model is None:
        raise RuntimeError("no surrogate trained yet -- run: python surrogate.py")
    margin = getattr(model, "meta", {}).get("margin", 100.0)

    combos = all_combos(inventory)
    total = len(combos)
    # Verify in blocks: big enough to keep every core busy, small enough that a
    # small inventory is not forced to open more than it needs to.
    block = block or int(min(10_000, max(2_000, total / 20)))
    if total == 0:
        return [], {}
    pred = model.predict(combos)
    order = np.argsort(-pred)

    best, done, t0, worst = [], 0, time.time(), 0.0
    with mp.Pool(processes or mp.cpu_count(), initializer=_init, initargs=(a_max,)) as pool:
        while done < total:
            take = order[done:done + block]
            evs = np.concatenate(list(pool.imap(_work_all, chunks(iter(combos[take])))))
            worst = max(worst, float((evs - pred[take]).max()))
            for ev, row in zip(evs, take):
                item = (float(ev), tuple(int(i) for i in combos[row]))
                if len(best) < top:
                    heapq.heappush(best, item)
                elif item > best[0]:
                    heapq.heapreplace(best, item)
            done += len(take)
            if progress:
                progress(done, total, time.time() - t0)
            if should_stop and should_stop():
                break
            # best is a min-heap of the top `top`, so best[0][0] is the worst
            # entry currently on the board.  Stopping when the best remaining
            # prediction cannot reach *that* keeps the whole ranking safe, not
            # just the winner.
            if (len(best) == top and done < total
                    and pred[order[done]] + margin < best[0][0]):
                break

    # The sweep is its own audit: every loadout it opened is a fresh test of the
    # model, so if any of them beat its prediction by more than the margin, the
    # calibration was optimistic and the run says so instead of staying quiet.
    stats = {"evaluated": done, "total": total, "margin": margin,
             "worst_underestimate": worst, "margin_held": worst <= margin,
             "fraction": done / total, "elapsed": time.time() - t0}
    return [(ev, [NAMES[i] for i in c]) for ev, c in sorted(best, reverse=True)], stats


def describe(combo, a_max=engine.DEFAULT_A_MAX):
    """Extra stats for one loadout, computed only for the handful we display."""
    e = engine.Engine(a_max=a_max)
    return {"turn_ev": e.turn_ev(combo), "bust_prob": e.bust_prob(combo)}


if __name__ == "__main__":
    import sys

    inv = {n: 1 for n in NAMES}
    if len(sys.argv) > 1:
        inv = {n: 1 for n in NAMES[:int(sys.argv[1])]}
    total = count_combos(inv)
    print(f"{len(inv)} die types -> {total:,} combinations on {mp.cpu_count()} cores")

    def show(done, tot, el):
        if done % (CHUNK * 50) < CHUNK or done == tot:
            rate = done / el if el else 0
            print(f"\r  {done:,}/{tot:,}  {done/tot*100:5.1f}%  "
                  f"{rate:,.0f}/s  eta {(tot-done)/rate if rate else 0:5.0f}s", end="")

    res = search(inv, top=10, progress=show)
    print(f"\ndone in {time.time():.0f}\n")
    for ev, combo in res:
        print(f"  {ev:9.2f}  {', '.join(combo)}")
