"""The 43 KCD2 dice, as exact integer weights.

The wiki (https://kingdom-come-deliverance.fandom.com/wiki/Dice/KCD2) publishes
rounded percentages; the game itself uses small integer weights.  Each vector
below was recovered by rational reconstruction -- the smallest denominator whose
faces reproduce every published percentage to within 0.1 -- and every one of the
43 dice reconstructs, worst residual 0.073 percentage points.  The wiki mixes
rounding and truncation (27.27 -> "27.2" but 9.375 -> "9.4"), which is why the
reconstruction accepts either.  Source percentages are in the comments.

Weights are over SEVEN values: (joker, 1, 2, 3, 4, 5, 6).

Two dice roll a joker face:
  Devil's head die -- the 1 is replaced by a devil's head
  Balatro's die    -- all six faces are jokers
A joker matches any combination but never scores on its own (see scoring.py).
"""

# name -> (weights over (joker,1,2,3,4,5,6), denominator)
DICE = {
    "Aranka's die": ((0, 6, 1, 6, 1, 6, 1), 21),  # 28.6%, 4.8%, 28.6%, 4.8%, 28.6%, 4.8%
    # Jimbo's grinning face replaces the 1, exactly like the Devil's head die.
    # (The wiki shows a joker on all six faces -- that is an unfilled template,
    # not the game's behaviour.)
    "Balatro's die": ((1, 0, 1, 1, 1, 1, 1), 6),
    "Cautious cheater's die": ((0, 5, 3, 2, 3, 5, 3), 21),  # 23.8%, 14.3%, 9.5%, 14.3%, 23.8%, 14.3%
    'Ci die': ((0, 3, 3, 3, 3, 3, 8), 23),  # 13%, 13%, 13%, 13%, 13%, 34.8%
    # the 1 face is a devil's head (joker); 16.7% each face
    "Devil's head die": ((1, 0, 1, 1, 1, 1, 1), 6),
    'Die of misfortune': ((0, 1, 5, 5, 5, 5, 1), 22),  # 4.5%, 22.7%, 22.7%, 22.7%, 22.7%, 4.5%
    'Even die': ((0, 1, 4, 1, 4, 1, 4), 15),  # 6.7%, 26.7%, 6.7%, 26.7%, 6.7%, 26.7%
    'Favourable die': ((0, 6, 0, 1, 1, 6, 4), 18),  # 33.3%, 0%, 5.6%, 5.6%, 33.3%, 22.2%
    'Fer die': ((0, 3, 3, 3, 3, 3, 8), 23),  # 13%, 13%, 13%, 13%, 13%, 34.8%
    'Greasy die': ((0, 3, 2, 3, 2, 3, 4), 17),  # 17.6%, 11.8%, 17.6%, 11.7%, 17.6%, 23.5%
    'Grimy die': ((0, 1, 5, 1, 1, 7, 1), 16),  # 6.2%, 31.2%, 6.2%, 6.2%, 43.7%, 6.2%
    "Grozav's lucky die": ((0, 1, 10, 1, 1, 1, 1), 15),  # 6.7%, 66.7%, 6.7%, 6.7%, 6.7%, 6.7%
    'Heavenly Kingdom die': ((0, 7, 2, 2, 2, 2, 4), 19),  # 36.8%, 10.5%, 10.5%, 10.5%, 10.5%, 21%
    'Holy Trinity die': ((0, 4, 5, 10, 1, 1, 1), 22),  # 18.2%, 22.7%, 45.4%, 4.5%, 4.5%, 4.5%
    "Hugo's die": ((0, 1, 1, 1, 1, 1, 1), 6),  # 16.7%, 16.7%, 16.7%, 16.7%, 16.7%, 16.7%
    "King's die": ((0, 4, 6, 7, 8, 4, 3), 32),  # 12.5%, 18.7%, 21.9%, 25%, 12.5%, 9.4%
    "Lousy gambler's die": ((0, 2, 3, 2, 3, 7, 3), 20),  # 10%, 15%, 10%, 15%, 35%, 15%
    'Lu die': ((0, 3, 3, 3, 3, 3, 8), 23),  # 13%, 13%, 13%, 13%, 13%, 34.8%
    'Lucky die': ((0, 6, 1, 2, 3, 4, 6), 22),  # 27.3%, 4.5%, 9.1%, 13.6%, 18.2%, 27.3%
    "Mathematician's die": ((0, 4, 5, 6, 7, 1, 1), 24),  # 16.7%, 20.8%, 25%, 29.2%, 4.2%, 4.2%
    'Molar die': ((0, 1, 1, 1, 1, 1, 1), 6),  # 16.7%, 16.7%, 16.7%, 16.7%, 16.7%, 16.7%
    "Monk's die": ((0, 8, 8, 1, 1, 1, 1), 20),  # 40%, 40%, 5%, 5%, 5%, 5%
    'Mother-of-pearl die': ((0, 3, 1, 1, 1, 3, 3), 12),  # 25%, 8.3%, 8.3%, 8.3%, 25%, 25%
    'Odd die': ((0, 4, 1, 4, 1, 4, 1), 15),  # 26.7%, 6.7%, 26.7%, 6.7%, 26.7%, 6.7%
    'Ordinary die': ((0, 1, 1, 1, 1, 1, 1), 6),  # 16.7%, 16.7%, 16.7%, 16.7%, 16.7%, 16.7%
    'Painted die': ((0, 3, 1, 1, 1, 7, 3), 16),  # 18.7%, 6.2%, 6.2%, 6.2%, 43.7%, 18.7%
    "Painter's die B": ((0, 1, 3, 2, 2, 2, 1), 11),  # 9.1%, 27.2%, 18.2%, 18.2%, 18.2%, 9.1%
    "Painter's die G": ((0, 1, 3, 2, 2, 2, 1), 11),  # 9.1%, 27.2%, 18.2%, 18.2%, 18.2%, 9.1%
    "Painter's die R": ((0, 1, 3, 2, 2, 2, 1), 11),  # 9.1%, 27.2%, 18.2%, 18.2%, 18.2%, 9.1%
    'Pie die': ((0, 6, 1, 3, 3, 0, 0), 13),  # 46.2%, 7.7%, 23.1%, 23.1%, 0%, 0%
    'Premolar die': ((0, 1, 1, 1, 1, 1, 1), 6),  # 16.7%, 16.7%, 16.7%, 16.7%, 16.7%, 16.7%
    "Sad Greaser's die": ((0, 6, 6, 1, 1, 6, 3), 23),  # 26.1%, 26.1%, 4.3%, 4.3%, 26.1%, 13%
    "Saint Antiochus' die": ((0, 3, 1, 6, 1, 1, 3), 15),  # 20%, 6.7%, 40%, 6.7%, 6.7%, 20%
    'Shrinking die': ((0, 2, 1, 1, 1, 1, 3), 9),  # 22.2%, 11.1%, 11.1%, 11.1%, 11.1%, 33.3%
    "St. Stephen's die": ((0, 1, 1, 1, 1, 1, 1), 6),  # 16.7%, 16.7%, 16.7%, 16.7%, 16.7%, 16.7%
    'Strip die': ((0, 4, 2, 2, 2, 3, 3), 16),  # 25%, 12.5%, 12.5%, 12.5%, 18.8%, 18.8%
    "Tengri's die": ((0, 2, 1, 1, 1, 1, 1), 7),  # 28.5%, 14.3%, 14.3%, 14.3%, 14.3%, 14.3%
    'Trinity die': ((0, 2, 1, 9, 1, 2, 1), 16),  # 12.5%, 6.2%, 56.2%, 6.2%, 12.5%, 6.2%
    'Unbalanced die': ((0, 3, 4, 1, 1, 2, 1), 12),  # 25%, 33.3%, 8.3%, 8.3%, 16.7%, 8.3%
    'Unlucky die': ((0, 1, 3, 2, 2, 2, 1), 11),  # 9.1%, 27.3%, 18.2%, 18.2%, 18.2%, 9.1%
    "Wagoner's die": ((0, 1, 5, 6, 2, 2, 2), 18),  # 5.6%, 27.8%, 33.3%, 11.1%, 11.1%, 11.1%
    'Weighted die': ((0, 10, 1, 1, 1, 1, 1), 15),  # 66.7%, 6.7%, 6.7%, 6.7%, 6.7%, 6.7%
    'Wisdom tooth die': ((0, 1, 1, 1, 1, 1, 1), 6),  # 16.7%, 16.7%, 16.7%, 16.7%, 16.7%, 16.7%
}

NAMES = sorted(DICE)
N_DICE = len(NAMES)
INDEX = {n: i for i, n in enumerate(NAMES)}

JOKER = 0  # index of the joker "face" in a weight vector


def probabilities(name):
    """Exact face probabilities of one die, as a 7-tuple over (joker, 1..6)."""
    w, n = DICE[name]
    return tuple(x / n for x in w)


def has_joker(name):
    return DICE[name][0][JOKER] > 0
