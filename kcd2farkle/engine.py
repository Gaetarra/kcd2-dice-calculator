"""Exact expected-value engine for a KCD2 Farkle turn.

State is (mask, a): which of the six dice are still on the table, and how many
points the turn has accumulated but not yet banked.

    V[mask][a] = E_roll[ max over holds (p, rest) of  max( bank a+p , V[rest][a+p] ) ]
    V[mask][a] = 0 on a bust -- the turn's unbanked points are lost

with rest == 0 meaning hot dice: all six are held, so you reroll all six and the
continuation is V[FULL][a+p].

Every hold scores at least 50, so *every* state on the right refers to a
strictly larger `a` -- the hot-dice edge back to V[FULL] included.  That makes
the whole thing a DAG: sweeping `a` downwards solves it in one backward pass,
with no fixed-point iteration and no need to order the masks.

Scores are stored in units of 50 points throughout.

Two objectives share the kernel:
  goal_idx == 0  maximise expected points banked this turn
  goal_idx > 0   maximise P(this turn banks at least goal_idx*50 points), which
                 is the same recursion with an indicator in place of the reward
"""

import numpy as np
from numba import njit

import tables
from dice_data import DICE, JOKER, probabilities

FULL = tables.FULL
N_MASK = 1 << tables.N_POS
DEFAULT_A_MAX = 12000  # ponytail: above this we assume you bank; test_engine.py bounds the error


@njit(cache=True, fastmath=True)
def _dp(fprob, cls_of, cls_opt_off, opt_pts, opt_base,
        mask_out_off, mask_cls_off, mask_pos, mask_k,
        n_face, n_acc, V, Vflat, stop_lut, prob, cprob,
        act_prob, act_lo, act_hi, act_off):
    # --- pass 1: probability of each outcome class, compacted ---
    # Only classes this loadout can actually roll survive, and bust classes are
    # dropped outright since they contribute nothing.  For a joker die that is
    # a 4x cut: the seven-face table has 34,398 classes but one joker in the set
    # leaves under 9,000 of them reachable.
    n = 0
    act_off[1] = 0
    for mask in range(1, N_MASK):
        k = mask_k[mask]
        prob[0] = 1.0
        size = 1
        for i in range(k):
            pos = mask_pos[mask, i]
            for f in range(n_face - 1, -1, -1):  # descending so f == 0 reads before it writes
                pf = fprob[pos, f]
                base = f * size
                for j in range(size):
                    prob[base + j] = pf * prob[j]
            size *= n_face
        c0 = mask_cls_off[mask]
        c1 = mask_cls_off[mask + 1]
        for c in range(c0, c1):
            cprob[c] = 0.0
        off_o = mask_out_off[mask]
        for o in range(size):
            pr = prob[o]
            if pr != 0.0:
                cprob[c0 + cls_of[off_o + o]] += pr
        for c in range(c0, c1):
            pr = cprob[c]
            if pr != 0.0 and cls_opt_off[c + 1] > cls_opt_off[c]:
                act_prob[n] = pr
                act_lo[n] = cls_opt_off[c]
                act_hi[n] = cls_opt_off[c + 1]
                n += 1
        act_off[mask + 1] = n

    # --- pass 2: backward sweep over the accumulated turn total ---
    # V is padded on the right with the "you must bank now" tail, so the inner
    # loop needs no bounds check and no branch on whether a hold clears the table.
    for ai in range(n_acc - 1, -1, -1):
        for mask in range(1, N_MASK):
            total = 0.0
            for j in range(act_off[mask], act_off[mask + 1]):
                best = 0.0  # a roll with no option at all is a bust, worth nothing
                for s in range(act_lo[j], act_hi[j]):
                    nxt = ai + opt_pts[s]
                    v = stop_lut[nxt]
                    roll = Vflat[opt_base[s] + nxt]
                    if roll > v:
                        v = roll
                    if v > best:
                        best = v
                total += act_prob[j] * best
            V[mask, ai] = total
    return V


MAX_PTS_IDX = 8000 // 50  # six 1s, the largest single hold


class Engine:
    """Holds the shared tables and the scratch buffers the kernel writes into."""

    def __init__(self, a_max=DEFAULT_A_MAX):
        self.n_acc = a_max // 50 + 1
        self.width = self.n_acc + MAX_PTS_IDX + 1
        self._t = {}
        self._buf = {}

    def _for(self, n_face):
        if n_face not in self._t:
            t = dict(tables.load(n_face))
            # a hold that clears the table is hot dice: continue from all six
            dst = np.where(t["opt_rem"] == 0, FULL, t["opt_rem"]).astype(np.int64)
            t["opt_base"] = dst * self.width
            self._t[n_face] = t
            n_cls = len(t["cls_opt_off"]) - 1
            self._buf[n_face] = (
                np.zeros((N_MASK, self.width)),
                np.zeros(n_face ** tables.N_POS),
                np.zeros(n_cls),
                np.zeros(n_cls),
                np.zeros(n_cls, np.int32),
                np.zeros(n_cls, np.int32),
                np.zeros(N_MASK + 1, np.int32),
            )
        return self._t[n_face], self._buf[n_face]

    def _stop_lut(self, goal_idx):
        i = np.arange(self.width)
        return (i >= goal_idx).astype(float) if goal_idx > 0 else i * 50.0

    def face_probs(self, combo):
        """(6, n_face) face probabilities; n_face is 7 only if a joker die is in play."""
        probs = [probabilities(d) for d in combo]
        joker = any(p[JOKER] > 0 for p in probs)
        n_face = tables.JOKER if joker else tables.PLAIN
        fp = np.zeros((tables.N_POS, n_face))
        for i, p in enumerate(probs):
            fp[i] = p if joker else p[1:]
        return fp, n_face

    def solve(self, combo, goal=0):
        """Value table for one six-die loadout. Rows are masks, columns are a/50."""
        fp, n_face = self.face_probs(combo)
        t, (V, prob, cprob, *act) = self._for(n_face)
        goal_idx = -(-goal // 50)
        if goal_idx >= self.n_acc:
            raise ValueError(f"goal {goal} exceeds a_max {(self.n_acc - 1) * 50}")
        stop_lut = self._stop_lut(goal_idx)
        V[:, self.n_acc:] = stop_lut[self.n_acc:]  # the tail: above the grid, bank
        _dp(fp, t["cls_of"], t["cls_opt_off"], t["opt_pts"], t["opt_base"],
            t["mask_out_off"], t["mask_cls_off"], t["mask_pos"], t["mask_k"],
            n_face, self.n_acc, V, V.ravel(), stop_lut, prob, cprob, *act)
        return V

    def turn_ev(self, combo):
        """Expected points banked in one turn, played optimally. The Tab 1 metric."""
        return self.solve(combo)[FULL, 0]

    def bust_prob(self, combo):
        """Chance the opening six-dice roll scores nothing."""
        fp, n_face = self.face_probs(combo)
        t, _ = self._for(n_face)
        off_o, off_c = t["mask_out_off"][FULL], t["mask_cls_off"][FULL]
        n_out = n_face ** tables.N_POS
        p = np.ones(1)
        for i in range(tables.N_POS):
            p = (fp[i][:, None] * p[None, :]).ravel()
        cls = off_c + t["cls_of"][off_o:off_o + n_out]
        return float(p[t["cls_opt_off"][cls + 1] == t["cls_opt_off"][cls]].sum())


    # ---- move evaluation (tab 2) -------------------------------------------

    def moves(self, combo, positions, faces, banked=0, goal=0, V=None):
        """Rank every distinct hold available from one roll.

        `positions` are the dice slots still on the table and `faces` what they
        just rolled (0 = joker).  Returns dicts sorted best-first, each with the
        value of banking immediately versus rolling on.
        """
        if V is None:
            V = self.solve(combo, goal)
        banked_idx = banked // 50
        goal_idx = -(-goal // 50)
        out, seen = [], set()
        for pts_idx, rem in tables._options_for(tuple(positions), tuple(faces)):
            nxt = banked_idx + pts_idx
            stop = (1.0 if nxt >= goal_idx else 0.0) if goal_idx > 0 else nxt * 50.0
            roll = V[FULL if rem == 0 else rem, nxt] if nxt < self.n_acc else stop
            held = [p for p in positions if not rem >> p & 1]
            # Two holds that take the same faces off the same *kinds* of die, and
            # leave the same kinds behind, are the same move -- don't list twice.
            key = (pts_idx,
                   tuple(sorted((combo[p], faces[positions.index(p)]) for p in held)),
                   tuple(sorted(combo[p] for p in positions if rem >> p & 1)))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "points": pts_idx * 50,
                "hold": held,
                "hold_faces": [faces[positions.index(p)] for p in held],
                "remaining": [p for p in positions if rem >> p & 1],
                "hot_dice": rem == 0,
                "bank_value": stop,
                "roll_value": roll,
                "value": max(stop, roll),
                "action": "roll" if roll > stop else "pass",
            })
        out.sort(key=lambda m: (-m["value"], -m["points"]))
        return out

    def best_move(self, combo, positions, faces, banked=0, goal=0, V=None):
        m = self.moves(combo, positions, faces, banked, goal, V)
        return m[0] if m else None


_DEFAULT = None


def default_engine():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Engine()
    return _DEFAULT


def turn_ev(combo):
    return default_engine().turn_ev(combo)
