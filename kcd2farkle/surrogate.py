"""An NNUE-shaped surrogate for turn EV, to make the huge searches fast.

A six-die loadout is a *multiset*, so the natural first layer is NNUE's: one
embedding row per die type, summed.  That is permutation-invariant by
construction -- the model cannot even represent an ordering -- and because the
sum is linear it collapses to a single matmul against a count vector, which is
how a few million loadouts get scored in seconds.

    accum = sum of W0[die] over the six dice        (equivalently counts @ W0)
    pred  = w2 . relu(W1 . relu(accum + b0) + b1) + b2

About 14,000 parameters, so it trains on the CPU in a minute and needs no GPU and
no deep-learning framework -- plain NumPy and Adam.

Labels come from engine.py, which is exact, so training data is free apart from
the time to generate it.

The point of the model is *pruning*, and a learned filter cannot prove it did
not discard the winner.  So calibrate() measures the error on held-out loadouts
the model never saw and reports a margin; search keeps everything within that
margin of the incumbent best.  That is a statistical guarantee, not a proof --
exact mode remains the proof, and stays the default.
"""

import os

import numpy as np

from dice_data import NAMES, N_DICE

H1, H2 = 128, 64
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_surrogate.npz")


def counts(combos):
    """(n, 6) die indices -> (n, 43) how many of each die type."""
    combos = np.asarray(combos)
    out = np.zeros((len(combos), N_DICE), np.float32)
    np.add.at(out, (np.arange(len(combos))[:, None], combos), 1.0)
    return out


class Surrogate:
    def __init__(self, params=None):
        self.p = params

    # ---- inference ---------------------------------------------------------

    def predict(self, combos, chunk=200_000):
        """Predicted turn EV. Chunked so a multi-million sweep stays in memory."""
        p = self.p
        out = np.empty(len(combos), np.float32)
        for i in range(0, len(combos), chunk):
            block = combos[i:i + chunk]
            a = np.maximum(counts(block) @ p["W0"] + p["b0"], 0.0)
            h = np.maximum(a @ p["W1"] + p["b1"], 0.0)
            out[i:i + chunk] = (h @ p["w2"] + p["b2"]) * p["scale"] + p["mean"]
        return out

    # ---- training ----------------------------------------------------------

    def fit(self, combos, y, epochs=600, batch=4096, lr=3e-3, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        X = counts(combos)
        mean, scale = float(y.mean()), float(y.std()) or 1.0
        t = ((y - mean) / scale).astype(np.float32)

        def he(shape, fan_in):
            return (rng.normal(0, np.sqrt(2.0 / fan_in), shape)).astype(np.float32)

        p = {"W0": he((N_DICE, H1), 6), "b0": np.zeros(H1, np.float32),
             "W1": he((H1, H2), H1), "b1": np.zeros(H2, np.float32),
             "w2": he(H2, H2), "b2": np.zeros((), np.float32)}
        m = {k: np.zeros_like(v) for k, v in p.items()}
        v = {k: np.zeros_like(v) for k, v in p.items()}
        step = 0

        for epoch in range(epochs):
            # cosine decay: the last epochs are what settle the top of the ranking
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
            order = rng.permutation(len(X))
            loss_sum = 0.0
            for i in range(0, len(X), batch):
                idx = order[i:i + batch]
                xb, tb = X[idx], t[idx]
                n = len(idx)

                z0 = xb @ p["W0"] + p["b0"]
                a0 = np.maximum(z0, 0.0)
                z1 = a0 @ p["W1"] + p["b1"]
                a1 = np.maximum(z1, 0.0)
                out = a1 @ p["w2"] + p["b2"]

                err = out - tb
                loss_sum += float(err @ err)
                g = (2.0 / n) * err
                gw2 = a1.T @ g
                gb2 = g.sum()
                ga1 = np.outer(g, p["w2"])
                gz1 = ga1 * (z1 > 0)
                gW1 = a0.T @ gz1
                gb1 = gz1.sum(0)
                ga0 = gz1 @ p["W1"].T
                gz0 = ga0 * (z0 > 0)
                gW0 = xb.T @ gz0
                gb0 = gz0.sum(0)
                grads = {"W0": gW0, "b0": gb0, "W1": gW1, "b1": gb1, "w2": gw2, "b2": gb2}

                step += 1
                for k in p:
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    mh = m[k] / (1 - 0.9 ** step)
                    vh = v[k] / (1 - 0.999 ** step)
                    p[k] -= lr_t * mh / (np.sqrt(vh) + 1e-8)

            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                rmse = np.sqrt(loss_sum / len(X)) * scale
                print(f"    epoch {epoch:4d}  train rmse {rmse:8.2f} EV")

        p["mean"] = np.float32(mean)
        p["scale"] = np.float32(scale)
        self.p = p
        return self

    # ---- calibration -------------------------------------------------------

    def calibrate(self, combos, y):
        """Error on loadouts the model never trained on.

        `margin` is what search must allow: a loadout is only safe to discard if
        its prediction plus the margin is still below the best EV found so far.
        """
        pred = self.predict(combos)
        residual = y - pred  # positive means the model under-rated this loadout
        return {
            "n": len(y),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "mae": float(np.mean(np.abs(residual))),
            "max_under": float(residual.max()),
            "q_9999": float(np.quantile(residual, 0.9999)),
            "margin": float(max(residual.max() * 2.0, 25.0)),
        }

    # ---- persistence -------------------------------------------------------

    def save(self, path=MODEL, meta=None):
        np.savez(path, **self.p, **{f"meta_{k}": v for k, v in (meta or {}).items()})

    @classmethod
    def load(cls, path=MODEL):
        if not os.path.exists(path):
            return None
        with np.load(path) as z:
            p = {k: z[k] for k in z.files if not k.startswith("meta_")}
            meta = {k[5:]: float(z[k]) for k in z.files if k.startswith("meta_")}
        s = cls(p)
        s.meta = meta
        return s


# ---- training data ---------------------------------------------------------

def sample_combos(n, rng, inventory=None, near=None, near_frac=0.35):
    """Random loadouts to label.

    Uniform sampling alone teaches the model about mediocre loadouts, which is
    not where pruning decisions get made.  So a share of the sample is drawn by
    perturbing known-good loadouts, which puts training data where the top of
    the ranking actually lives.
    """
    pool = np.array(sorted({NAMES.index(k) for k in inventory} if inventory
                           else range(N_DICE)))
    out = rng.choice(pool, size=(n, 6))
    if near is not None and len(near):
        k = int(n * near_frac)
        seeds = np.asarray(near)[rng.integers(0, len(near), k)]
        swaps = rng.integers(0, 6, (k, 2))
        for j in range(2):
            seeds[np.arange(k), swaps[:, j]] = rng.choice(pool, k)
        out[:k] = seeds
    return np.sort(out, axis=1).astype(np.int8)


def build(n_train=200_000, n_calib=40_000, seed=0, epochs=600, progress=print):
    """Generate labels with the exact engine, train, calibrate, save."""
    import search

    rng = np.random.default_rng(seed)
    progress("sampling uniform loadouts...")
    warm = sample_combos(n_train // 3, rng)
    warm_y = search.evaluate(warm)
    best = warm[np.argsort(warm_y)[-2000:]]

    progress(f"labelling {n_train + n_calib:,} loadouts with the exact engine...")
    rest = sample_combos(n_train + n_calib - len(warm), rng, near=best)
    rest_y = search.evaluate(rest)
    X = np.concatenate([warm, rest])
    y = np.concatenate([warm_y, rest_y])

    cut = len(X) - n_calib
    progress(f"training on {cut:,}, calibrating on {n_calib:,}")
    model = Surrogate().fit(X[:cut], y[:cut], epochs=epochs, seed=seed)
    stats = model.calibrate(X[cut:], y[cut:])
    model.save(meta=stats)
    progress("calibration: " + "  ".join(f"{k}={v:,.2f}" for k, v in stats.items()))
    return model, stats


if __name__ == "__main__":
    import sys
    import time

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    t0 = time.time()
    build(n_train=n, n_calib=max(5000, n // 4))
    print(f"total {time.time() - t0:.0f}s")
