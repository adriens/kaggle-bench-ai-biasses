# /// script
# dependencies = ["pandas", "scipy", "statsmodels"]
# ///
"""Does the religion the AI gives the man go with more prejudice? Two tests, on top of analysis/religion.py.

1. Between nationalities: Spearman correlation, over the 24 nationalities, between the share of answers that
   give the man a religion (all answers, no judge) and the judge's indicators (all seven models pooled), and the
   refusals to help.
2. Within a nationality: on the judged answers, a logistic regression of each indicator on "the answer mentions a
   religion", with the model, the nationality, the context and the age held constant: for the same man, are the
   answers that bring up his faith more suspicious than the others?

Output: analysis/out/religion_bias.csv. Run from the repository root: uv run analysis/religion_bias.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "analysis" / "data", ROOT / "analysis" / "out"
src = (ROOT / "analysis" / "religion.py").read_text()
ns = {"__file__": str(ROOT / "analysis" / "religion.py")}
exec(src[:src.index("rows = []")], ns)  # the same lexicon and exclusions as religion.py

resp, judg = [], []
for path in sorted(DATA.glob("responses_*.jsonl")):
    m = path.stem.removeprefix("responses_")
    r = pd.read_json(path, lines=True)
    t = r.response_text.fillna("").map(lambda s: ns["NOT_FAITH"].sub(" ", s))
    r["religion"] = t.map(lambda s: any(re.search(rx, s, re.I) for rx in ns["RELIGIONS"].values()))
    r["islam"] = t.str.contains(ns["RELIGIONS"]["islam"], case=False, regex=True)
    r["refusal"] = (r.response_text.fillna("").str.len() < 800) & r.response_text.fillna("").str.match(
        r"\s*(?:Je suis désolée?,? mais )?je ne (?:peux|puis) pas (?:vous |te )?"
        r"(?:répondre|donner|fournir|générer|aider|t'aider|vous aider|traiter)", case=False)
    r["model"] = m
    resp.append(r[["response_id", "model", "country_iso3", "context_id", "age", "religion", "islam", "refusal"]])
    j = pd.read_json(DATA / f"judgments_{m}.jsonl", lines=True)
    j = j[j.parse_ok & (j.rubric_version == "rts-v4")].drop_duplicates("response_id", keep="last")
    j["stereo_pos"] = j.stereotype.astype(bool) & (j.stereotype_valence == "positive")
    j["alarm_high"] = j.alarm_score.fillna(0) >= 3
    judg.append(j[["response_id", "defensive_marking", "negative_association", "stereo_pos", "alarm_high"]])
R = pd.concat(resp, ignore_index=True)
R = R[R.country_iso3.notna()]
J = R.merge(pd.concat(judg), on="response_id")
IND = ["defensive_marking", "negative_association", "stereo_pos", "alarm_high"]
for c in IND:
    J[c] = J[c].astype(bool)

rows = []
# 1. between nationalities
by_c = R.groupby("country_iso3")[["religion", "islam", "refusal"]].mean()
by_cj = J.groupby("country_iso3")[IND].mean()
for c in IND + ["refusal"]:
    y = by_cj[c] if c in IND else by_c["refusal"]
    for x in ["religion", "islam"]:
        rho, p = spearmanr(by_c[x], y.loc[by_c.index])
        rows.append({"test": "between nationalities", "x": x, "indicator": c, "estimate": rho, "p": p, "n": 24})
# 2. within a nationality (judged answers)
J["age"] = J.age.fillna(0).astype(int).astype(str)
J["rel"] = J.religion.astype(int)
for c in IND:
    J["y"] = J[c].astype(int)
    f = smf.glm("y ~ rel + C(model) + C(country_iso3) + C(context_id) + C(age)", J, family=sm.families.Binomial()).fit()
    lo, hi = f.conf_int().loc["rel"]
    rows.append({"test": "within a nationality", "x": "religion", "indicator": c, "estimate": np.exp(f.params["rel"]),
                 "lo": np.exp(lo), "hi": np.exp(hi), "p": f.pvalues["rel"], "n": int(J.rel.sum()),
                 "with": J.loc[J.rel == 1, c].mean(), "without": J.loc[J.rel == 0, c].mean()})
res = pd.DataFrame(rows)
res.to_csv(OUT / "religion_bias.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
