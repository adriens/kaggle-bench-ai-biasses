# Alone With a Stranger

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23991568.svg)](https://doi.org/10.5281/zenodo.23991568)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Report: *She Is Alone With a Man* (`report/report.pdf`), with a 7-page version (`report/short.pdf`), a
16-slide carousel (`report/carousel.pdf`) and 4 square images (`report/square.pdf`), deposited on Zenodo:
[doi.org/10.5281/zenodo.23991568](https://doi.org/10.5281/zenodo.23991568) (all versions; version 1.1 is [doi.org/10.5281/zenodo.23992096](https://doi.org/10.5281/zenodo.23992096)). Notebooks and datasets: the
[Kaggle collection](https://www.kaggle.com/work/collections/19371544).

Benchmark of nationality bias in the safety advice that open-weight language models give to a woman
alone with a man. Same French prompt, only the man's nationality changes (24 nationalities + a control),
across 11 contexts, 3 paraphrases, 3 ages and 3 samples; answers rated by an LLM judge (Qwen3-32B-AWQ,
rubric `rts-v4`).

| Path | Content |
|---|---|
| `PLAN.md` | protocol, decisions and run log (French) |
| `kaggle/` | generation bench (`country_bias_bench.py`), judge (`judge.py`), Kaggle push tooling (`push.py`) |
| `analysis/` | answers and verdicts (`data/`), Python synthesis notebook, exported figures |
| `report/` | scientific report: Quarto + R, compiled with XeLaTeX (`report.pdf`) |
| `covers/` | early cover visuals |

## Rebuild the report

```sh
cd report
QUARTO_R=/usr/bin/Rscript quarto render report.qmd   # delete data.rds first if the data changes
Rscript R/covers.R                                   # social-media cover proposals
```

## How to cite

Sales, A. (2026). *She Is Alone With a Man: Nationality bias in the safety advice given by open-weight
language models* (Version 1.1) [Report]. Zenodo. https://doi.org/10.5281/zenodo.23992096

## Versions

Each version deposited on Zenodo has a git tag with the same number, on the commit whose PDFs are the ones
deposited: [`v1.0`](https://github.com/adriens/kaggle-bench-ai-biasses/releases/tag/v1.0)
([10.5281/zenodo.23991569](https://doi.org/10.5281/zenodo.23991569)),
[`v1.1`](https://github.com/adriens/kaggle-bench-ai-biasses/releases/tag/v1.1)
([10.5281/zenodo.23992096](https://doi.org/10.5281/zenodo.23992096)).

The documents and the data are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the
Kaggle notebooks are under Apache 2.0, Kaggle's licence for public notebooks.

Author: Adrien Sales · [x.com/rastadidi](https://x.com/rastadidi) · [dev.to/adriens](https://dev.to/adriens) ·
[linkedin.com/in/adrien-sales](https://www.linkedin.com/in/adrien-sales) ·
[kaggle.com/adriensales](https://www.kaggle.com/adriensales)
