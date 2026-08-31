"""
Extract feature lists from features.xlsx.

Three flavors are produced (F11 is always excluded — it is the OUD label code):

  1. Inclusive (OR-ranked, no clinical curation filter):
       top{50,100,150,200,300}_features.txt
     These were used in the main 8-model × 5-feature-set experiments.

  2. Kelly limited (49 codes): rows where the KELLY ET column is BLANK,
     ordered by OR. Kelly's accepted subset is small and clinically strict.
     Output: kelly_features.txt

  3. Rick expanded (252 codes): rows where the RICK ET column is BLANK,
     ordered by OR. Rick's accepted subset is the broader clinical list.
     Output: rick_top{50,100,150,200,252}_features.txt
     (rick_top252 = the full Rick list; top-300 is skipped because Rick
      only has 252 codes.)

Run with:
    python src/01_extract_features.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR


def _write_list(name: str, codes: list[str]):
    out_path = OUT_DIR / f"{name}_features.txt"
    out_path.write_text("\n".join(codes))
    print(f"  saved {out_path}  ({len(codes)} codes)")
    if codes:
        print(f"    first 5: {codes[:5]}")
        print(f"    last  5: {codes[-5:]}")


def main():
    df = pd.read_excel(DATA_DIR / "features.xlsx")
    print(f"Total features in spreadsheet: {len(df)}")
    print(f"Row 0 (excluded - label code): {df.iloc[0]['ICD10']} — "
          f"{df.iloc[0]['DIAGNOSIS_DESCRIPTION*']}")

    # Everything below excludes F11 (row 0)
    body = df.iloc[1:].copy()
    or_ranked = body["ICD10"].tolist()
    print(f"Usable features after excluding F11: {len(or_ranked)}")

    # ── 1. Inclusive (OR-ranked, no curation filter) ──────────────────────
    print("\n[inclusive lists — OR-ranked, no clinician filter]")
    for n in [50, 100, 150, 200, 300]:
        _write_list(f"top{n}", or_ranked[:n])

    # ── 2. Kelly limited — rows where KELLY ET is blank ───────────────────
    kelly_mask = body["KELLY ET"].isna()
    kelly_codes = body.loc[kelly_mask, "ICD10"].tolist()
    print(f"\n[Kelly limited — rows where KELLY ET is blank]"
          f"  size = {len(kelly_codes)}")
    _write_list("kelly", kelly_codes)

    # ── 3. Rick expanded — rows where RICK ET is blank ────────────────────
    rick_mask = body["RICK ET"].isna()
    rick_codes = body.loc[rick_mask, "ICD10"].tolist()
    print(f"\n[Rick expanded — rows where RICK ET is blank]"
          f"  size = {len(rick_codes)}")
    rick_sizes = [50, 100, 150, 200, len(rick_codes)]
    seen = set()
    for n in rick_sizes:
        # rick_top{n} where n = len(rick_codes) is "rick_top252"
        # (the full Rick list); name uses the actual size for clarity.
        if n in seen:
            continue
        seen.add(n)
        _write_list(f"rick_top{n}", rick_codes[:n])


if __name__ == "__main__":
    main()
