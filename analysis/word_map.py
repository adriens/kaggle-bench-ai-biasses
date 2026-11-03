# /// script
# dependencies = ["pandas", "scikit-learn", "simplemma"]
# ///
"""Counts, for each nationality, how often the AI uses each of its most specific words, for the map of men and
words (correspondence analysis) that opens the report.

The words are the three most over-used lemmas of each nationality (Lafon specificities from the guess-the-
nationality notebook, analysis/out/guess/specificities.csv), minus words that say nothing about the man. They are
counted in every answer with a nationality, of every model, after the judge's masking; the lemmatisation and the
function words are those of the guess-the-nationality notebook, read from analysis/guess_nationality.py so that
the two cannot drift apart. Output: analysis/out/word_map_counts.csv (nationality x lemma counts).

Run from the repository root: uv run analysis/word_map.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
import simplemma
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "analysis" / "data"
OUT = ROOT / "analysis" / "out" / "word_map_counts.csv"
PER_COUNTRY = 3
NOT_A_TRAIT = {"personne", "pays", "nationalité", "europe", "européen", "amérique", "user", "time", "translate",
               "google", "diversifier", "généralement", "association", "relation", "gendarmerie", "numéro", "local",
               "qualité", "petit", "région", "concept", "dating", "traducteur", "langue", "culturel", "culture",
               "valeur", "structure", "espace",
               "test", "utiliser", "meeting",  # procedure or English, not the man
               "ponctuel", "débattre"}  # same idea as ponctualité and débat, already in

# masking: the judge's own code, as in defensive_regex.py
judge = (ROOT / "kaggle" / "build" / "judge" / "judge.py").read_text()
masking = judge[judge.index("def inflections"):judge.index('print(mask("Je suis')]
masking = masking.replace('pd.read_csv(next(iter(BENCH.values())) / "countries.csv")',
                          f'pd.read_csv(r"{DATA / "countries.csv"}")')
# lemmas: the guess-the-nationality notebook's own code (function words, lemma(), lemmas())
guess = (ROOT / "analysis" / "guess_nationality.py").read_text()
leak = guess[guess.index("LEAK = set("):guess.index("def leaky")]
lex = guess[guess.index("STOP = set("):guess.index("LEM = d.shared.map(lemmas)")]
ns = {"re": re, "pd": pd, "sys": sys, "simplemma": simplemma, "ENGLISH_STOP_WORDS": ENGLISH_STOP_WORDS}
exec(masking, ns)
exec(leak, ns)
exec(lex, ns)

spec = pd.read_csv(ROOT / "analysis" / "out" / "guess" / "specificities.csv")
spec = spec[~spec.lemma.isin(NOT_A_TRAIT) & (spec.specificity > 0)]
words = sorted(set(spec.sort_values("specificity", ascending=False).groupby("country").head(PER_COUNTRY).lemma))
print(len(words), "words:", ", ".join(words))

rows = []
for path in sorted(DATA.glob("responses_*.jsonl")):
    r = pd.read_json(path, lines=True)[["country_iso3", "response_text"]].dropna(subset=["country_iso3"])
    for c, text in zip(r.country_iso3, r.response_text.fillna("")):
        lem = ns["lemmas"](ns["mask"](text))
        rows.append(pd.Series(lem, dtype=object).value_counts().reindex(words, fill_value=0).rename(c))
    print(path.stem, len(r))
counts = pd.DataFrame(rows).groupby(level=0).sum()
OUT.parent.mkdir(parents=True, exist_ok=True)
counts.to_csv(OUT)
print("written", OUT, counts.shape)
