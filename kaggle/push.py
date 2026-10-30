"""Push the bench notebook to Kaggle as one kernel per model.

A new version of a Kaggle kernel replaces the outputs of the previous one: one kernel per model keeps every
model's results. The model is injected into the config cell of country_bias_bench.py, the notebook is built
with jupytext under build/<key>/, then pushed.

    python push.py granite-4.2-8b             # build and push
    python push.py granite-4.2-8b --dry-run   # build only
    python push.py granite-4.2-8b --resume    # attach the previous output, to resume an unfinished run
    python push.py --list

    python push.py --judge granite-4.2-8b [...]           # judge kernel, reading these bench kernels
    python push.py --judge granite-4.2-8b --resume        # + its own previous verdicts, to finish a run
    python push.py --judge granite-4.2-8b --speed-test    # 200 verdicts per judge variant, to compare cost
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

KAGGLE_USER = "adriensales"
HERE = Path(__file__).parent
SOURCE = HERE / "country_bias_bench.py"
JUDGE_SOURCE = HERE / "judge.py"
SPEED_TEST_N = 200

# multimodal models may need {"limit_mm_per_prompt": {...}} to save memory: to tune at the smoke test
MODELS = {
    "granite-4.2-8b": {"model_id": "ibm-granite/granite-4.2-8b"},
    "qwen3.5-9b": {"model_id": "Qwen/Qwen3.5-9B"},
    "gemma-4-e4b-it": {"model_id": "google/gemma-4-E4B-it"},
    "ministral-3-8b": {"model_id": "mistralai/Ministral-3-8B-Instruct-2512"},
    "llama-3.2-3b": {"model_id": "meta-llama/Llama-3.2-3B-Instruct"},
    "llama-3.1-8b": {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
    "gemma-4-31b-awq": {"model_id": "QuantTrio/gemma-4-31B-it-AWQ", "quantization": "awq"},
    "mistral-small-3.2-24b-awq": {
        "model_id": "jeffcookio/Mistral-Small-3.2-24B-Instruct-2506-awq-sym",
        "quantization": "awq",
    },
}


TITLE = "Alone With a Stranger"


def kernel_title(key):
    return f"{TITLE} - {key}"


def kernel_slug(key):
    # Kaggle derives the kernel slug from its title: both must match
    return re.sub(r"[^a-z0-9]+", "-", kernel_title(key).lower()).strip("-")


def set_config(source, name, value):
    pattern = re.compile(rf"^{name} = .*$", re.MULTILINE)
    assert len(pattern.findall(source)) == 1, f"{name} must be assigned exactly once in {SOURCE.name}"
    return pattern.sub(f"{name} = {value!r}", source)


def kernel_metadata(slug, title, code_file, kernel_sources):
    return {
        "id": f"{KAGGLE_USER}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
        "model_sources": [],
    }


def build(key, resume):
    model = MODELS[key]
    source = SOURCE.read_text()
    source = set_config(source, "MODEL_ID", model["model_id"])
    source = set_config(source, "QUANTIZATION", model.get("quantization"))
    source = set_config(source, "LLM_EXTRA_ARGS", model.get("llm_extra_args", {}))
    # the notebook's heading names the model too
    assert source.count("\n# # Country Bias Bench\n") == 1
    source = source.replace("\n# # Country Bias Bench\n", f"\n# # {TITLE} — {model['model_id']}\n")

    out = HERE / "build" / key
    out.mkdir(parents=True, exist_ok=True)
    (out / SOURCE.name).write_text(source)
    subprocess.run(["jupytext", "--to", "ipynb", "-q", str(out / SOURCE.name)], check=True)

    slug = kernel_slug(key)
    # the kernel's own previous output, mounted under /kaggle/input to resume
    sources = [f"{KAGGLE_USER}/{slug}"] if resume else []
    metadata = kernel_metadata(slug, kernel_title(key), SOURCE.with_suffix(".ipynb").name, sources)
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return out


def build_judge(keys, resume, speed_test):
    """The judge kernel, with the outputs of the given bench kernels attached as inputs."""
    title = kernel_title("judge")
    slug = kernel_slug("judge")
    out = HERE / "build" / "judge"
    out.mkdir(parents=True, exist_ok=True)
    source = JUDGE_SOURCE.read_text()
    if speed_test:
        source = set_config(source, "SPEED_TEST_N", SPEED_TEST_N)
    (out / JUDGE_SOURCE.name).write_text(source)
    subprocess.run(["jupytext", "--to", "ipynb", "-q", str(out / JUDGE_SOURCE.name)], check=True)
    sources = [f"{KAGGLE_USER}/{kernel_slug(k)}" for k in keys] + ([f"{KAGGLE_USER}/{slug}"] if resume else [])
    metadata = kernel_metadata(slug, title, JUDGE_SOURCE.with_suffix(".ipynb").name, sources)
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="model key(s); one, except with --judge")
    parser.add_argument("--dry-run", action="store_true", help="build without pushing")
    parser.add_argument("--resume", action="store_true", help="attach the kernel's previous output as input")
    parser.add_argument("--list", action="store_true", help="list the models and their kernels")
    parser.add_argument("--judge", action="store_true", help="build the judge kernel for these models' outputs")
    parser.add_argument("--speed-test", action="store_true", help=f"judge only: {SPEED_TEST_N} verdicts per variant")
    args = parser.parse_args()

    if args.list or not args.models:
        for key, model in MODELS.items():
            print(f"{key:28} {model['model_id']:58} {KAGGLE_USER}/{kernel_slug(key)}")
        return
    unknown = set(args.models) - set(MODELS)
    if unknown:
        parser.error(f"unknown model(s): {', '.join(sorted(unknown))} (see --list)")
    if not args.judge and len(args.models) > 1:
        parser.error("one model per bench kernel (several only with --judge)")

    if args.speed_test and not args.judge:
        parser.error("--speed-test goes with --judge")

    out = build_judge(args.models, args.resume, args.speed_test) if args.judge else build(args.models[0], args.resume)
    print("built", out)
    if not args.dry_run:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(out)], check=True)


if __name__ == "__main__":
    main()
