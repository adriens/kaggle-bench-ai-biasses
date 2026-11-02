# /// script
# dependencies = ["pandas"]
# ///
"""Checks the judge's defensive marking against the bench's own rule (a regular expression on the masked answer).

The masking and the rule are read from the judge notebook (kaggle/build/judge/judge.py) and run unchanged, so the
three copies cannot drift apart. Output: analysis/out/defensive_regex.csv, one row per model and nationality
(BASE for the control), with the judge's and the rule's share of answers and their agreement, answer by answer.

Run from the repository root: uv run analysis/defensive_regex.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "analysis" / "data"
OUT = ROOT / "analysis" / "out" / "defensive_regex.csv"
MODELS = ["granite-4.2-8b", "qwen3.5-9b", "gemma-4-e4b-it", "ministral-3-8b-instruct-2512",
          "nvidia-nemotron-3.5-bolt-e4b-bf16"]

src = (ROOT / "kaggle" / "build" / "judge" / "judge.py").read_text()
masking = src[src.index("def inflections"):src.index('print(mask("Je suis')]
masking = masking.replace('pd.read_csv(next(iter(BENCH.values())) / "countries.csv")',
                          f'pd.read_csv(r"{DATA / "countries.csv"}")')
end = src.index("    return False", src.index("def defensive")) + len("    return False")
rule = src[src.index("DEFENSIVE_RE = re.compile"):end]
ns = {"re": re, "pd": pd}
exec(masking, ns)
exec(rule, ns)

rows = []
for m in MODELS:
    j = pd.read_json(DATA / f"judgments_{m}.jsonl", lines=True)
    j = j[j.parse_ok & (j.rubric_version == "rts-v4")].drop_duplicates("response_id", keep="last")
    r = pd.read_json(DATA / f"responses_{m}.jsonl", lines=True)[["response_id", "country_iso3", "response_text"]]
    d = j.merge(r, on="response_id")
    d["judge"] = d.defensive_marking.astype(bool)
    d["regex"] = d.response_text.map(lambda t: ns["defensive"](ns["mask"](t)))
    d["country"] = d.country_iso3.fillna("BASE")
    g = d.groupby("country").agg(n=("judge", "size"), judge=("judge", "sum"), regex=("regex", "sum"),
                                 both=("judge", lambda s: int((s & d.loc[s.index, "regex"]).sum())))
    g.insert(0, "model", m)
    rows.append(g.reset_index())
    print(f"{m:30} judge {d.judge.mean():6.1%}  regex {d.regex.mean():6.1%}")

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.concat(rows).to_csv(OUT, index=False)
print("written", OUT)
