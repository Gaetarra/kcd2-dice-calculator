"""KCD2 Farkle calculator -- two tabs, tkinter.

Tab 1  Best Dice Combination: exhaustive exact search over your inventory.
Tab 2  Best Move: what to hold, and whether to roll on or pass.
"""

import multiprocessing as mp
import queue
import threading
import tkinter as tk
from tkinter import ttk

import engine
import search
from dice_data import DICE, NAMES, has_joker, probabilities

POLL_MS = 80
MS_PER_COMBO = 0.0015  # measured on 20 threads; only used for the eta hint


def pct(name):
    p = probabilities(name)
    return "".join(f"{p[f] * 100:6.1f}" for f in range(1, 7)) + ("   J" if has_joker(name) else "")


class App(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=8)
        self.pack(fill="both", expand=True)
        self.engine = engine.Engine()
        self.queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.worker = None
        self.last_stats = None

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.tab1 = ttk.Frame(nb, padding=8)
        self.tab2 = ttk.Frame(nb, padding=8)
        nb.add(self.tab1, text="  Best Dice Combination  ")
        nb.add(self.tab2, text="  Best Move  ")
        self._build_tab1()
        self._build_tab2()
        self.after(POLL_MS, self._poll)

    # ---------------- tab 1 -------------------------------------------------

    def _build_tab1(self):
        left = ttk.Frame(self.tab1)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Dice you own", font="-weight bold").pack(anchor="w")
        header = "qty  " + "die".ljust(24) + "".join(f"{f:>6}" for f in range(1, 7))
        ttk.Label(left, text=header, font="TkFixedFont").pack(anchor="w")

        canvas = tk.Canvas(left, width=540, height=420, highlightthickness=0)
        bar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-e.delta // 120, "units"))

        self.qty = {}
        for row, name in enumerate(NAMES):
            var = tk.IntVar(value=0)
            var.trace_add("write", self._update_count)
            self.qty[name] = var
            ttk.Spinbox(inner, from_=0, to=6, width=2, textvariable=var,
                        command=self._update_count).grid(row=row, column=0, sticky="w", padx=(0, 6))
            ttk.Label(inner, text=name.ljust(24) + pct(name), font="TkFixedFont").grid(
                row=row, column=1, sticky="w")

        bottom = ttk.Frame(self.tab1)
        bottom.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(bottom, text="One of each", command=lambda: self._set_all(1)).pack(side="left")
        ttk.Button(bottom, text="Clear", command=lambda: self._set_all(0)).pack(side="left", padx=4)
        self.count_lbl = ttk.Label(bottom, text="")
        self.count_lbl.pack(side="left", padx=12)
        self.mode = tk.StringVar(value="exact")
        ttk.Radiobutton(bottom, text="exact", variable=self.mode, value="exact",
                        command=self._update_count).pack(side="left", padx=(8, 0))
        ttk.Radiobutton(bottom, text="fast", variable=self.mode, value="fast",
                        command=self._update_count).pack(side="left")
        self.go_btn = ttk.Button(bottom, text="Search", command=self._start)
        self.go_btn.pack(side="left", padx=4)
        self.cancel_btn = ttk.Button(bottom, text="Cancel", command=self.stop_flag.set,
                                     state="disabled")
        self.cancel_btn.pack(side="left")
        self.prog = ttk.Progressbar(bottom, length=200, mode="determinate")
        self.prog.pack(side="left", padx=8)
        self.status = ttk.Label(bottom, text="")
        self.status.pack(side="left")

        right = ttk.Frame(self.tab1)
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))
        ttk.Label(right, text="Best loadouts  (double-click one to load it into Best Move)",
                  font="-weight bold").pack(anchor="w")
        self.results = ttk.Treeview(right, columns=("ev", "bust", "dice"), show="headings",
                                    height=22)
        for col, width, title in (("ev", 90, "turn EV"), ("bust", 70, "bust %"),
                                  ("dice", 520, "six dice")):
            self.results.heading(col, text=title)
            self.results.column(col, width=width, anchor="w" if col == "dice" else "e")
        self.results.pack(fill="both", expand=True)
        self.results.bind("<Double-1>", self._load_into_tab2)
        self._update_count()

    def _set_all(self, n):
        for v in self.qty.values():
            v.set(n)

    def inventory(self):
        return {n: v.get() for n, v in self.qty.items() if v.get() > 0}

    def _update_count(self, *_):
        inv = self.inventory()
        total = search.count_combos(inv) if inv else 0
        est = total * MS_PER_COMBO
        eta = f"   ~{est / 60:.0f} min" if est > 90 else (f"   ~{est:.0f}s" if est > 2 else "")
        note = ""
        if self.mode.get() == "fast":
            import surrogate
            model = surrogate.Surrogate.load()
            note = ("   fast: ranked by the surrogate, then verified downwards until the "
                    f"margin ({getattr(model, 'meta', {}).get('margin', 0):,.0f} EV) rules the rest out"
                    if model else "   fast: no model trained yet -- run  python surrogate.py")
            eta = ""
        self.count_lbl.config(text=f"{total:,} combinations{eta}{note}")
        self.go_btn.config(state="normal" if total else "disabled")

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        inv = self.inventory()
        total = search.count_combos(inv)
        self.results.delete(*self.results.get_children())
        self.stop_flag.clear()
        self.go_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.prog.config(value=0, maximum=total)

        fast = self.mode.get() == "fast"

        def run():
            try:
                report = lambda d, t, e: self.queue.put(("progress", (d, t, e)))
                if fast:
                    res, stats = search.search_fast(
                        inv, top=25, progress=report, should_stop=self.stop_flag.is_set)
                    self.queue.put(("stats", stats))
                else:
                    res = search.search(
                        inv, top=25, progress=report, should_stop=self.stop_flag.is_set)
                self.queue.put(("done", res))
            except Exception as exc:  # surface worker failures instead of hanging the UI
                self.queue.put(("error", exc))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    done, total, elapsed = payload
                    self.prog.config(value=done)
                    rate = done / elapsed if elapsed else 0
                    left = (total - done) / rate if rate else 0
                    self.status.config(
                        text=f"{done:,}/{total:,}   {done / total * 100:.1f}%   "
                             f"{rate:,.0f}/s   eta {left // 60:.0f}m{left % 60:02.0f}s")
                elif kind == "stats":
                    self.last_stats = payload
                elif kind == "done":
                    self._show_results(payload)
                elif kind == "error":
                    self.status.config(text=f"failed: {payload}")
                    self.go_btn.config(state="normal")
                    self.cancel_btn.config(state="disabled")
        except queue.Empty:
            pass
        self.after(POLL_MS, self._poll)

    def _show_results(self, res):
        self.go_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        if self.stop_flag.is_set():
            note = "cancelled -- results only cover what was checked"
        elif self.mode.get() == "fast" and getattr(self, "last_stats", None):
            st = self.last_stats
            note = (f"done in {st['elapsed']:.0f}s -- exactly evaluated {st['evaluated']:,} of "
                    f"{st['total']:,} ({st['fraction'] * 100:.2f}%); the rest fell below the "
                    f"best found by more than the {st['margin']:,.0f} EV calibrated margin")
            if not st["margin_held"]:
                note += (f"   WARNING: the model under-rated an opened loadout by "
                         f"{st['worst_underestimate']:,.0f}, above its own margin -- "
                         f"retrain, or use exact mode")
        else:
            note = "done -- exhaustive, every combination exactly evaluated"
        self.status.config(text=note)
        for ev, combo in res:
            self.results.insert("", "end", values=(
                f"{ev:.2f}", f"{self.engine.bust_prob(combo) * 100:.2f}", ", ".join(combo)))

    def _load_into_tab2(self, _event):
        sel = self.results.focus()
        if not sel:
            return
        names = self.results.item(sel, "values")[2].split(", ")
        for var, name in zip(self.slot, names):
            var.set(name)
        self._refresh_faces()

    # ---------------- tab 2 -------------------------------------------------

    def _build_tab2(self):
        top = ttk.LabelFrame(self.tab2, text="Your six dice, and what they just rolled", padding=8)
        top.pack(fill="x")
        self.slot, self.on_table, self.face = [], [], []
        for i in range(6):
            ttk.Label(top, text=f"die {i + 1}").grid(row=i, column=0, sticky="w", pady=1)
            name = tk.StringVar(value="Ordinary die")
            box = ttk.Combobox(top, textvariable=name, values=NAMES, width=26, state="readonly")
            box.grid(row=i, column=1, padx=4)
            box.bind("<<ComboboxSelected>>", self._refresh_faces)
            keep = tk.BooleanVar(value=True)
            ttk.Checkbutton(top, text="on table", variable=keep,
                            command=self._refresh_faces).grid(row=i, column=2, padx=6)
            face = ttk.Combobox(top, width=7, state="readonly")
            face.grid(row=i, column=3)
            self.slot.append(name)
            self.on_table.append(keep)
            self.face.append(face)

        ctl = ttk.Frame(self.tab2)
        ctl.pack(fill="x", pady=8)
        ttk.Label(ctl, text="Points banked this turn").pack(side="left")
        self.banked = tk.StringVar(value="0")
        ttk.Entry(ctl, textvariable=self.banked, width=8).pack(side="left", padx=(4, 16))
        ttk.Label(ctl, text="Points still needed to win (0 = maximise EV)").pack(side="left")
        self.needed = tk.StringVar(value="0")
        ttk.Entry(ctl, textvariable=self.needed, width=8).pack(side="left", padx=4)
        ttk.Button(ctl, text="Analyse", command=self._analyse).pack(side="left", padx=12)

        self.verdict = ttk.Label(self.tab2, text="", font="-size 11 -weight bold", wraplength=1100)
        self.verdict.pack(anchor="w", pady=(4, 2))
        self.detail = ttk.Label(self.tab2, text="", foreground="#555", wraplength=1100)
        self.detail.pack(anchor="w", pady=(0, 6))

        self.moves = ttk.Treeview(
            self.tab2, columns=("hold", "pts", "then", "bank", "roll", "value"),
            show="headings", height=14)
        for col, width, title in (("hold", 300, "hold these"), ("pts", 70, "scores"),
                                  ("then", 70, "then"), ("bank", 120, "value if you pass"),
                                  ("roll", 120, "value if you roll on"), ("value", 120, "best")):
            self.moves.heading(col, text=title)
            self.moves.column(col, width=width, anchor="w" if col == "hold" else "e")
        self.moves.pack(fill="both", expand=True)
        self._refresh_faces()

    def _refresh_faces(self, *_):
        for i in range(6):
            weights, _ = DICE[self.slot[i].get()]
            allowed = [("joker" if f == 0 else str(f)) for f in range(7) if weights[f] > 0]
            box = self.face[i]
            box.config(values=allowed, state="readonly" if self.on_table[i].get() else "disabled")
            if box.get() not in allowed:
                box.set(allowed[0])

    def _analyse(self):
        combo = [v.get() for v in self.slot]
        positions = [i for i in range(6) if self.on_table[i].get()]
        if not positions:
            self.verdict.config(text="Hot dice: nothing left on the table, so you roll all six.")
            self.detail.config(text="")
            self.moves.delete(*self.moves.get_children())
            return
        faces = [0 if self.face[i].get() == "joker" else int(self.face[i].get())
                 for i in positions]
        try:
            banked = max(0, int(self.banked.get() or 0)) // 50 * 50
            needed = max(0, int(self.needed.get() or 0))
        except ValueError:
            self.verdict.config(text="Points must be whole numbers.")
            return
        if banked // 50 >= self.engine.n_acc or needed // 50 >= self.engine.n_acc:
            self.verdict.config(text=f"Points must be under {(self.engine.n_acc - 1) * 50}.")
            return

        V = self.engine.solve(combo, goal=needed)
        moves = self.engine.moves(combo, positions, faces, banked, needed, V)
        self.moves.delete(*self.moves.get_children())
        if not moves:
            self.verdict.config(
                text=f"BUST -- nothing in this roll scores, so the {banked} points "
                     "from this turn are lost.")
            self.detail.config(text="")
            return

        goal_mode = needed > 0
        fmt = (lambda v: f"{v * 100:.2f}%") if goal_mode else (lambda v: f"{v:,.1f}")
        for m in moves:
            self.moves.insert("", "end", values=(
                self._describe(m["hold"], m["hold_faces"], combo), m["points"],
                m["action"].upper(), fmt(m["bank_value"]), fmt(m["roll_value"]),
                fmt(m["value"])))

        best = moves[0]
        if best["action"] == "pass":
            tail = "then PASS and bank it"
        elif best["hot_dice"]:
            tail = "then ROLL ON -- that is hot dice, you get all six back"
        else:
            tail = f"then ROLL ON with the other {len(best['remaining'])}"
        self.verdict.config(
            text=f"Hold  {self._describe(best['hold'], best['hold_faces'], combo)}"
                 f"   (+{best['points']}),  {tail}.")
        self.detail.config(text=(
            f"turn total becomes {banked + best['points']}       "
            f"passing is worth {fmt(best['bank_value'])},  rolling on is worth "
            f"{fmt(best['roll_value'])}"
            + (f"       (maximising the chance of reaching {needed} this turn)"
               if goal_mode else "")))

    @staticmethod
    def _describe(positions, faces, combo):
        if not positions:
            return "nothing"
        return ", ".join(
            ("joker" if f == 0 else str(f)) + f" ({combo[p].replace(' die', '')})"
            for p, f in zip(positions, faces))


if __name__ == "__main__":
    mp.freeze_support()
    root = tk.Tk()
    root.title("KCD2 Farkle calculator")
    root.geometry("1300x660")
    App(root)
    root.mainloop()
