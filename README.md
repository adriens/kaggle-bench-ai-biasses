# Alone With a Stranger

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

Author: Adrien Sales · [x.com/rastadidi](https://x.com/rastadidi) · [dev.to/adriens](https://dev.to/adriens) ·
[linkedin.com/in/adrien-sales](https://www.linkedin.com/in/adrien-sales) ·
[kaggle.com/adriensales](https://www.kaggle.com/adriensales)
