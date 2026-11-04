# /// script
# dependencies = ["pandas", "scikit-learn"]
# ///
"""What decides whether an answer is prejudiced? A random forest on the judged answers.

The target is a prejudiced answer: one that ties the man's origin to a risk (negative association), states a
negative stereotype, or says, unasked, that his origin does not matter (defensive marking). The features are the
factors of the design and what the AI does: who made the model and its size, the man's nationality (BASE for the control), the context,
the woman's age, and whether the answer gives the man a religion (analysis/religion.py, run it first). The
forest is fitted on 75 % of the answers; its ROC AUC and the permutation importance of each factor (the drop in
AUC when the factor is shuffled) are measured on the other 25 %. The decision tree of the report (R, rpart) reads
the same answers.

A logistic regression and a small neural network, fitted on the same split, are kept for comparison.

Outputs (analysis/out/): bias_forest_importance.csv, bias_forest_metrics.csv.
Run from the repository root: uv run analysis/bias_forest.py
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "analysis" / "data", ROOT / "analysis" / "out"
FEATURES = ["publisher", "size", "nationality", "context", "age", "religion"]
# what describes a model: who made (or post-trained) it, and its size in billions of parameters as in its name
# (for the mixture of experts Gemma 4 26B-A4B, 26 in all, 4 active per token)
PUBLISHER = {"granite-4.2-8b": "IBM", "qwen3.5-9b": "Alibaba", "gemma-4-e2b-it": "Google", "gemma-4-e4b-it": "Google",
             "gemma-4-26b-a4b-it-awq-4bit": "Google", "ministral-3-8b-instruct-2512": "Mistral AI",
             "nvidia-nemotron-3.5-bolt-e4b-bf16": "NVIDIA"}
SIZE = {"granite-4.2-8b": "8B", "qwen3.5-9b": "9B", "gemma-4-e2b-it": "2B", "gemma-4-e4b-it": "4B",
        "gemma-4-26b-a4b-it-awq-4bit": "26B", "ministral-3-8b-instruct-2512": "8B",
        "nvidia-nemotron-3.5-bolt-e4b-bf16": "4B"}

religion = set(pd.read_csv(OUT / "religion_answers.csv").response_id)
rows = []
for path in sorted(DATA.glob("responses_*.jsonl")):
    m = path.stem.removeprefix("responses_")
    r = pd.read_json(path, lines=True)[["response_id", "country_iso3", "context_id", "age"]]
    j = pd.read_json(DATA / f"judgments_{m}.jsonl", lines=True)
    j = j[j.parse_ok & (j.rubric_version == "rts-v4")].drop_duplicates("response_id", keep="last")
    d = j.merge(r, on="response_id")
    d["prejudiced"] = (d.negative_association.astype(bool) | d.defensive_marking.astype(bool)
                       | (d.stereotype.astype(bool) & (d.stereotype_valence == "negative"))).astype(int)
    rows.append(pd.DataFrame({
        "publisher": PUBLISHER[m], "size": SIZE[m], "nationality": d.country_iso3.fillna("BASE"), "context": d.context_id,
        "age": d.age.fillna(0).astype(int).astype(str), "religion": d.response_id.isin(religion).map({True: "yes", False: "no"}),
        "prejudiced": d.prejudiced}))
D = pd.concat(rows, ignore_index=True)
print(len(D), "judged answers,", f"{D.prejudiced.mean():.1%} prejudiced")

X_tr, X_te, y_tr, y_te = train_test_split(D[FEATURES], D.prejudiced, test_size=0.25, random_state=0,
                                          stratify=D.prejudiced)
forest = make_pipeline(
    ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)]),
    RandomForestClassifier(n_estimators=400, min_samples_leaf=10, random_state=0, n_jobs=-1))
forest.fit(X_tr, y_tr)
auc = roc_auc_score(y_te, forest.predict_proba(X_te)[:, 1])
imp = permutation_importance(forest, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=0, n_jobs=-1)
importance = pd.DataFrame({"factor": FEATURES, "auc_drop": imp.importances_mean, "sd": imp.importances_std})
importance = importance.sort_values("auc_drop", ascending=False)
importance.to_csv(OUT / "bias_forest_importance.csv", index=False)
# for comparison: a linear model and a small neural network on the same split
others = {}
for name, clf in [("logistic", LogisticRegression(max_iter=2000)),
                  ("neural_net", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                                               random_state=0))]:
    pipe = make_pipeline(ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)]), clf)
    others[name] = roc_auc_score(y_te, pipe.fit(X_tr, y_tr).predict_proba(X_te)[:, 1])
pd.DataFrame([{"n": len(D), "prejudiced": D.prejudiced.mean(), "auc": auc, "n_test": len(X_te),
               "auc_logistic": others["logistic"], "auc_neural_net": others["neural_net"]}]).to_csv(
    OUT / "bias_forest_metrics.csv", index=False)
print({k: round(v, 3) for k, v in others.items()})
print(f"held-out ROC AUC {auc:.3f}")
print(importance.round(4).to_string(index=False))
