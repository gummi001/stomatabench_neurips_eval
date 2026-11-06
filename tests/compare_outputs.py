"""Compare every CSV under a new output folder with the released result tables."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RTOL = 1e-9
ATOL = 1e-9


def first_difference(new, ref):
    for col in ref.columns:
        a, b = new[col], ref[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(float), b.to_numpy(float)
            same = np.isclose(x, y, rtol=RTOL, atol=ATOL) | (np.isnan(x) & np.isnan(y))
        else:
            same = (a.astype(str) == b.astype(str)).to_numpy()
        if not same.all():
            i = int(np.flatnonzero(~same)[0])
            return f"{col}: {int((~same).sum())} cells differ, first at row {i}: {a.iloc[i]!r} vs {b.iloc[i]!r}"
    return None


def compare(new_csv, ref_csv):
    """None if equal; rows may come in another order when the key columns identify them."""
    new = pd.read_csv(new_csv, low_memory=False)
    ref = pd.read_csv(ref_csv, low_memory=False)
    if list(new.columns) != list(ref.columns):
        return f"columns differ: {sorted(set(new.columns) ^ set(ref.columns))}"
    if len(new) != len(ref):
        return f"rows differ: {len(new)} vs {len(ref)}"
    problem = first_difference(new, ref)
    if problem is None:
        return None
    keys = [c for c in ref.columns if not pd.api.types.is_float_dtype(ref[c])]
    if keys and not ref.duplicated(keys).any():
        new = new.sort_values(keys, kind="mergesort").reset_index(drop=True)
        ref = ref.sort_values(keys, kind="mergesort").reset_index(drop=True)
        if first_difference(new, ref) is None:
            return None
    return problem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("new", type=Path, help="output folder written by scripts/run_eval.sh")
    ap.add_argument("ref", type=Path, help="reference folder, e.g. results/")
    a = ap.parse_args()

    refs = sorted(p.relative_to(a.ref) for p in a.ref.rglob("*") if p.name.endswith((".csv", ".csv.gz")))
    failed = missing = 0
    for rel in refs:
        new_csv = a.new / rel
        if not new_csv.exists():
            print(f"MISSING  {rel}")
            missing += 1
            continue
        problem = compare(new_csv, a.ref / rel)
        if problem:
            print(f"DIFFERS  {rel}: {problem}")
            failed += 1
        else:
            print(f"same     {rel}")
    print(f"\n{len(refs) - failed - missing} same, {failed} differ, {missing} missing, of {len(refs)} reference CSVs")
    return 1 if failed or missing else 0


if __name__ == "__main__":
    sys.exit(main())
