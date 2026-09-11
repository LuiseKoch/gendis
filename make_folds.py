"""Regenerate the fold assignment from the raw annotation files.

AUTHORS ONLY, and rarely needed. The released dataset already carries the
fold column, so `prepare_data.py` reproduces the paper's splits exactly;
that is the normal way to build data/processed.csv.

This script is the original provenance of those folds. It reads the raw
per-country annotation files, which are not part of the public release, and
so cannot run without them.

Note that it overwrites data/processed.csv with the base columns only. The
stratum, language and per-annotator columns that `prepare_data.py` carries
through are dropped, which leaves eval_random_stratum.py, eval_by_language.py
and iaa.py unable to run until prepare_data.py is run again."""

from gendis import PROCESSED, assign_folds, countries, load_data


df = load_data()
df = assign_folds(df)

keep = ["idx", "country", "text", "gold_l1", "gold_l2", "gold4", "fold",
        "keyword_present"]
keep = [c for c in keep if c in df.columns]
df[keep].to_csv(str(PROCESSED), index=False)

print("rows:", len(df), "| countries:", countries(df))
print("\nfold sizes:\n", df["fold"].value_counts().sort_index())
print("\ngold4 x fold:\n", df.pivot_table(index="gold4", columns="fold",
      values="idx", aggfunc="count", fill_value=0))
print("\nwrote processed.csv")
