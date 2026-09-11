"""Step 1. Build the shared fold assignment and the canonical processed table.

Run this ONCE. Every other script reads processed.csv so that all systems are
trained and evaluated on the identical comparable partition."""

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
