# /// script
# dependencies = ["pandas", "scipy", "requests"]
# ///
"""Does the man's treatment follow the wealth of his country? A test across the 24 nationalities.

Wealth comes from the World Bank (CC BY 4.0): GDP per capita in current US$ (indicator NY.GDP.PCAP.CD) for one
fixed year, and the income group of each country (low, lower-middle, upper-middle, high; the classification
current on the day of the download, not that of the year), both from the World Bank open-data API. They are fetched once and kept in analysis/data/worldbank.csv, so that the test can be replayed
offline on the same figures.

For each nationality, the judged answers of the seven models are pooled (each model answers each nationality the
same number of times): the share of answers that say, unasked, that his origin does not matter (defensive
marking), that compliment his culture (positive stereotype), and that tie him to a risk (negative association).
Each share is set against GDP per capita with a Spearman rank correlation, over 24 points.

Outputs (analysis/out/): wealth_by_country.csv, wealth_spearman.csv.
Run from the repository root: uv run analysis/wealth.py
"""
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "analysis" / "data", ROOT / "analysis" / "out"
WB = DATA / "worldbank.csv"
YEAR = 2023  # the latest year with GDP per capita for all 24 countries at the time of the study
API = "https://api.worldbank.org/v2"

countries = pd.read_csv(DATA / "countries.csv").country_iso3.tolist()

if not WB.exists():
    codes = ";".join(countries)
    gdp = requests.get(f"{API}/country/{codes}/indicator/NY.GDP.PCAP.CD",
                       params={"format": "json", "date": YEAR, "per_page": 100}, timeout=60).json()[1]
    meta = requests.get(f"{API}/country/{codes}", params={"format": "json", "per_page": 100}, timeout=60).json()[1]
    wb = pd.DataFrame({"country": [g["countryiso3code"] for g in gdp], "gdp_pc": [g["value"] for g in gdp]}).merge(
        pd.DataFrame({"country": [m["id"] for m in meta], "income_group": [m["incomeLevel"]["value"] for m in meta]}),
        on="country")
    wb.insert(1, "year", YEAR)
    wb.to_csv(WB, index=False)
wb = pd.read_csv(WB)

rows = []
for path in sorted(DATA.glob("responses_*.jsonl")):
    m = path.stem.removeprefix("responses_")
    r = pd.read_json(path, lines=True)[["response_id", "country_iso3"]]
    j = pd.read_json(DATA / f"judgments_{m}.jsonl", lines=True)
    j = j[j.parse_ok & (j.rubric_version == "rts-v4")].drop_duplicates("response_id", keep="last")
    d = j.merge(r, on="response_id").dropna(subset=["country_iso3"])
    rows.append(pd.DataFrame({
        "country": d.country_iso3, "defensive": d.defensive_marking.astype(bool),
        "compliment": d.stereotype.astype(bool) & (d.stereotype_valence == "positive"),
        "risk": d.negative_association.astype(bool)}))
rates = (pd.concat(rows).groupby("country").mean() * 100).reset_index()
by = rates.merge(wb, on="country")
by.to_csv(OUT / "wealth_by_country.csv", index=False)

tests = []
for k in ["defensive", "compliment", "risk"]:
    rho, p = spearmanr(by.gdp_pc, by[k])
    tests.append({"measure": k, "rho": rho, "p": p, "n": len(by)})
pd.DataFrame(tests).to_csv(OUT / "wealth_spearman.csv", index=False)

print(by.sort_values("gdp_pc", ascending=False).round(1).to_string(index=False))
print(pd.DataFrame(tests).round(4).to_string(index=False))
