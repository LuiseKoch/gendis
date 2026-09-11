"""Inter-annotator agreement, per country and overall.

Three annotators (A1/A2/A3). We report Fleiss' kappa and Krippendorff's
alpha for Level 1, Level 2, and the combined 4-way label. Krippendorff's
alpha is the headline because it tolerates the missing Level-2 ratings that
arise when an annotator marked a comment as not gendered.

These numbers are the HUMAN CEILING that model scores are compared against.
"""
import numpy as np
import pandas as pd
from common import load_data, countries

try:
    import krippendorff
    HAVE_KRIPP = True
except ImportError:
    HAVE_KRIPP = False

try:
    from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
    HAVE_FLEISS = True
except ImportError:
    HAVE_FLEISS = False

L1_COLS = ["A1_level1", "A2_level1", "A3_level1"]
L2_COLS = ["A1_level2", "A2_level2", "A3_level2"]


def fleiss(ratings: pd.DataFrame):
    """ratings: rows=items, cols=annotators, integer categories (no NaN)."""
    r = ratings.dropna().astype(int)
    if len(r) < 2:
        return np.nan
    table, _ = aggregate_raters(r.values)
    return fleiss_kappa(table)


def kripp(ratings: pd.DataFrame):
    """Krippendorff alpha, nominal; tolerates NaN (missing ratings)."""
    if not HAVE_KRIPP:
        return np.nan
    data = ratings.T.values.astype(float)  # reliability_data: raters x units
    return krippendorff.alpha(reliability_data=data,
                              level_of_measurement="nominal")


def combined(df):
    """4-way label per annotator: 0=notgendered else level2 (shifted)."""
    out = {}
    for a in ["A1", "A2", "A3"]:
        l1 = df[f"{a}_level1"]
        l2 = df[f"{a}_level2"]
        # not gendered -> category 0; gendered -> level2 (+1) in {1,2,3}
        out[a] = np.where(l1 == 0, 0, l2 + 1)
    return pd.DataFrame(out)


def report(df, name):
    print(f"\n=== {name}  (N={len(df)}) ===")
    l1 = df[L1_COLS]
    l2 = df[L2_COLS]
    cmb = combined(df)
    rows = [
        ("Level 1 (gendered/not)", fleiss(l1), kripp(l1)),
        ("Level 2 (neg/neu/pos)",  fleiss(l2), kripp(l2)),
        ("Combined 4-way",         fleiss(pd.DataFrame(cmb)), kripp(pd.DataFrame(cmb))),
    ]
    print(f"{'task':<26}{'Fleiss k':>10}{'Kripp a':>10}")
    for t, fk, ka in rows:
        print(f"{t:<26}{fk:>10.3f}{ka:>10.3f}")


if __name__ == "__main__":
    if not HAVE_FLEISS:
        print("WARN: statsmodels missing -> Fleiss kappa unavailable")
    if not HAVE_KRIPP:
        print("WARN: krippendorff missing -> alpha unavailable (pip install krippendorff)")
    df = load_data()
    report(df, "ALL COUNTRIES")
    for c in countries(df):
        report(df[df.country == c], c)
