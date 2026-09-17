"""Checks for the KCD2 Farkle engine.  Run: python test_engine.py"""

import re
import sys
import time

import numpy as np

import dice_data
import engine
import tables
from scoring import hold_options, score_hold


def counts(*faces, jokers=0):
    v = [jokers] + [0] * 6
    for f in faces:
        v[f] += 1
    return tuple(v)


def test_scoring():
    assert score_hold(counts(1, 2, 3, 4, 5, 6)) == 1500, "full straight"
    assert score_hold(counts(2, 3, 4, 5, 6)) == 750, "straight 2-6"
    assert score_hold(counts(1, 2, 3, 4, 5)) == 500, "straight 1-5"
    assert score_hold(counts(2, 2, 2, 2, 2, 2)) == 1600, "six of a kind = 200 * 8"
    assert score_hold(counts(1, 1, 1)) == 1000
    assert score_hold(counts(1, 1, 1, 1)) == 2000, "quad beats triple + single"
    assert score_hold(counts(1, 2, 3, 4, 5, 5)) == 550, "straight 1-5 plus a loose 5"
    assert score_hold(counts(1)) == 100 and score_hold(counts(5)) == 50
    assert score_hold(counts(2, 2, 3, 4, 6)) is None, "must use every held die"
    assert score_hold(counts(2)) is None
    # the wiki's three worked joker examples
    assert score_hold(counts(1, 1, jokers=1)) == 1000, "two 1s + devil = triple"
    assert score_hold(counts(1, 1, 1, jokers=1)) == 2000, "three 1s + devil = quad"
    assert score_hold(counts(1, 2, 3, 4, jokers=1)) == 500, "1234 + devil = straight 1-5"
    assert score_hold(counts(jokers=1)) is None, "a joker never scores on its own"
    assert score_hold(counts(5, jokers=1)) is None, "nor does it prop up a lone 5"
    assert hold_options(counts(2, 2, 3, 4, 6)) == (), "bust"
    print("  scoring ok")


def test_weights():
    """Every stored weight vector must reproduce the percentages the wiki publishes."""
    src = open(dice_data.__file__, encoding="utf-8").read()
    checked = 0
    for name, (w, n) in dice_data.DICE.items():
        m = re.search(re.escape(repr(name)) + r".*?#\s*([\d.%,\s]+)$", src, re.M)
        if not m:
            continue
        published = [float(x.strip().rstrip("%")) for x in m.group(1).split(",")]
        for face, pct in enumerate(published, start=1):
            got = w[face] / n * 100
            assert abs(got - pct) < 0.1005, f"{name} face {face}: {got:.2f} vs {pct}"
        assert sum(w) == n, f"{name} weights must sum to the denominator"
        checked += 1
    assert checked == 41, f"expected 41 ordinary dice, checked {checked}"
    assert len(dice_data.DICE) == 43
    assert sorted(n for n in dice_data.NAMES if dice_data.has_joker(n)) == [
        "Balatro's die", "Devil's head die"]
    print(f"  weights ok ({checked} dice against published percentages)")


def test_bust_probability():
    """Six fair dice: no 1, no 5, no triple.  Countable by hand."""
    e = engine.Engine()
    got = e.bust_prob(["Ordinary die"] * 6)
    exact = 1440 / 6 ** 6  # (2,2,2,0) and (2,2,1,1) splits over the four blank faces
    assert abs(got - exact) < 1e-12, f"{got} vs {exact}"
    print(f"  bust probability ok ({got:.6f} == 1440/46656)")


def test_tables_match_brute_force():
    """Enumerate rolls directly, bypassing tables.py, to catch an indexing slip.

    Six fair dice cannot catch one -- every position is interchangeable -- so
    these cases are deliberately asymmetric, including the joker dice.
    """
    import itertools

    e = engine.Engine()
    for combo in (["Ordinary die"] * 6,
                  ["Weighted die", "Pie die", "Odd die", "King's die",
                   "Grimy die", "Favourable die"],
                  ["Balatro's die", "Devil's head die", "Weighted die",
                   "Pie die", "Odd die", "Ordinary die"]):
        fps = [dice_data.probabilities(d) for d in combo]
        bust = 0.0
        for faces in itertools.product(*[[f for f in range(7) if fps[i][f] > 0]
                                         for i in range(6)]):
            c = [0] * 7
            for f in faces:
                c[f] += 1
            if not hold_options(tuple(c)):
                p = 1.0
                for i, f in enumerate(faces):
                    p *= fps[i][f]
                bust += p
        got = e.bust_prob(combo)
        assert abs(bust - got) < 1e-12, f"{combo}: brute {bust} vs engine {got}"
    print("  roll tables match brute-force enumeration")


def test_jit_matches_python():
    """The compiled kernel and the plain-Python one must agree exactly."""
    e = engine.Engine(a_max=400)
    combo = ["Ordinary die", "Weighted die", "Odd die", "Pie die", "Lucky die", "Even die"]
    fp, n_face = e.face_probs(combo)
    t, (V, prob, cprob, *act) = e._for(n_face)
    stop_lut = e._stop_lut(0)
    args = (t["cls_of"], t["cls_opt_off"], t["opt_pts"], t["opt_base"],
            t["mask_out_off"], t["mask_cls_off"], t["mask_pos"], t["mask_k"],
            n_face, e.n_acc)
    V[:, e.n_acc:] = stop_lut[e.n_acc:]
    jit = engine._dp(fp, *args, V, V.ravel(), stop_lut, prob, cprob, *act).copy()
    V2 = np.zeros_like(V)
    V2[:, e.n_acc:] = stop_lut[e.n_acc:]
    pure = engine._dp.py_func(fp, *args, V2, V2.ravel(), stop_lut,
                              np.zeros_like(prob), np.zeros_like(cprob),
                              *[np.zeros_like(a) for a in act]).copy()
    assert np.allclose(jit, pure, atol=1e-12), np.abs(jit - pure).max()
    print("  jit == pure python ok")


def test_monotone_and_truncation():
    combo = ["Weighted die", "Favourable die", "Lucky die", "Odd die", "King's die", "Grimy die"]
    e = engine.Engine()
    V = e.solve(combo)
    # Truncating the grid puts a downward step at its very top -- above a_max we
    # force a bank, which is worth less than rolling on would have been.  Below
    # a_max minus the largest single hold, no state can reach that step.
    safe = e.n_acc - engine.MAX_PTS_IDX
    for mask in range(1, 64):
        d = np.diff(V[mask, :safe])
        assert (d >= -1e-9).all(), f"V[{mask}] must be non-decreasing in the turn total"
    assert V[engine.FULL, 0] > 0
    wide = engine.Engine(a_max=20000).turn_ev(combo)
    assert abs(V[engine.FULL, 0] - wide) < 0.01, f"truncation error {V[engine.FULL,0]-wide}"
    print(f"  monotone below {safe*50} pts; truncation error "
          f"{abs(V[engine.FULL,0]-wide):.4f} points")


def simulate(e, combo, n, seed=0):
    """Play n turns with the policy the DP itself recommends."""
    V = e.solve(combo)
    fp = np.array([dice_data.probabilities(d) for d in combo])
    rng = np.random.default_rng(seed)
    faces_by_pos = [np.flatnonzero(fp[p]) for p in range(6)]
    probs_by_pos = [fp[p][faces_by_pos[p]] for p in range(6)]
    out = np.empty(n)
    for i in range(n):
        positions, banked = list(range(6)), 0
        while True:
            faces = [int(rng.choice(faces_by_pos[p], p=probs_by_pos[p])) for p in positions]
            m = e.best_move(combo, positions, faces, banked, V=V)
            if m is None:
                banked = 0
                break
            banked += m["points"]
            if m["action"] == "pass" or banked // 50 >= e.n_acc:
                break
            positions = m["remaining"] or list(range(6))
        out[i] = banked
    return out


def test_dp_matches_simulation():
    """The number the DP reports must be the number its own policy actually scores."""
    e = engine.Engine()
    for combo, n in ((["Ordinary die"] * 6, 40000),
                     (["Weighted die", "Odd die", "Lucky die",
                       "Ordinary die", "Even die", "Pie die"], 40000)):
        exact = e.turn_ev(combo)
        t0 = time.time()
        sample = simulate(e, combo, n)
        se = sample.std(ddof=1) / np.sqrt(n)
        z = abs(sample.mean() - exact) / se
        print(f"  simulated {n} turns in {time.time()-t0:4.1f}s: "
              f"mean {sample.mean():8.2f} +/- {se:5.2f}  vs exact {exact:8.2f}  ({z:.2f} sigma)")
        assert z < 4, f"DP and simulation disagree by {z:.1f} sigma"


def test_fast_matches_exact():
    """Fast mode must return the same ranking the exhaustive search does."""
    import search
    import surrogate

    model = surrogate.Surrogate.load()
    if model is None:
        print("  skipped -- no surrogate trained (run: python surrogate.py)")
        return
    inv = {n: 1 for n in dice_data.NAMES[:18]}
    exact = search.search(inv, top=10)
    fast, stats = search.search_fast(inv, top=10)
    assert [c for _, c in exact] == [c for _, c in fast], (
        f"fast mode disagreed: exact {exact[0]} vs fast {fast[0]}")
    print(f"  fast == exact on {stats['total']:,} loadouts; "
          f"opened {stats['evaluated']:,} ({stats['fraction']*100:.2f}%), "
          f"margin {stats['margin']:,.0f} EV")


if __name__ == "__main__":
    for fn in [test_scoring, test_weights, test_bust_probability,
               test_tables_match_brute_force, test_jit_matches_python,
               test_monotone_and_truncation, test_dp_matches_simulation,
               test_fast_matches_exact]:
        print(fn.__name__)
        fn()
    print("\nall checks passed")
