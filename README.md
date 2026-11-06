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

## Resources

**Report** (Zenodo, CC BY 4.0): [doi.org/10.5281/zenodo.23991568](https://doi.org/10.5281/zenodo.23991568), all
versions, always the latest; [v1.1](https://doi.org/10.5281/zenodo.23992096), [v1.0](https://doi.org/10.5281/zenodo.23991569).

**Kaggle datasets** (CC BY 4.0)

- [Alone With a Stranger - analysis data](https://www.kaggle.com/datasets/adriensales/alone-with-a-stranger-analysis-data):
  51,975 answers of the 7 models and the judge's verdicts
- [Alone With a Stranger - World Bank wealth](https://www.kaggle.com/datasets/adriensales/alone-with-a-stranger-world-bank-wealth):
  GDP per capita and income group of the 24 countries

**Kaggle notebooks** (Apache 2.0), all in the [Alone With a Stranger collection](https://www.kaggle.com/work/collections/19371544)

| Notebook | Role |
|---|---|
| [granite-4.2-8b](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-granite-4-2-8b) | bench, Granite 4.2 8B (IBM) |
| [qwen3.5-9b](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-qwen3-5-9b) | bench, Qwen3.5-9B (Alibaba) |
| [gemma-4-e2b-it](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-gemma-4-e2b-it) | bench, Gemma 4 E2B (Google) |
| [gemma-4-e4b-it](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-gemma-4-e4b-it) | bench, Gemma 4 E4B (Google) |
| [gemma-4-26b-a4b-awq](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-gemma-4-26b-a4b-awq) | bench, Gemma 4 26B-A4B, 4-bit AWQ (Google) |
| [ministral-3-8b](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-ministral-3-8b) | bench, Ministral 3 8B (Mistral AI) |
| [nemotron-3.5-bolt-e4b](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-nemotron-3-5-bolt-e4b) | bench, Nemotron 3.5 Bolt E4B (NVIDIA) |
| [gemma-4-31b-awq](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-gemma-4-31b-awq) | bench attempt, Gemma 4 31B AWQ: out of memory on 2 × T4, not in the report |
| [judge](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-judge) | LLM judge (Qwen3-32B-AWQ, rubric `rts-v4`) |
| [synthese-des-juges](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-synthese-des-juges) | synthesis of the verdicts, figures (French) |
| [guess-the-nationality](https://www.kaggle.com/code/adriensales/alone-with-a-stranger-guess-the-nationality) | judge-free check: guess the nationality from the masked answers |

**Data sources**: [RTS report](https://www.rts.ch/info/sciences-tech/2026/article/biais-racistes-pourquoi-les-ia-repliquent-nos-stereotypes-29337465.html)
and [France Inter debate](https://www.radiofrance.fr/franceinter/podcasts/le-debat/le-debat-de-la-grand-matinale-du-jeudi-03-septembre-2026-5351193)
(the starting point); [World Bank GDP per capita](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD) (CC BY 4.0).

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
