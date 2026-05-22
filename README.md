# Local Coder Benchmark

LM Studio benchmark harness for evaluating local models as coding workers,
bug-fixers, and technical handoff summarizers.

The benchmark dynamically discovers downloaded LM Studio LLMs with `lms ls`,
loads each model through the `lms` CLI, runs the test suite through LM Studio's
OpenAI-compatible API, unloads the model, and saves a ranked report.

## What It Tests

- `HumanEval`: Python function completion from docstrings and signatures.
- `MBPP`: short Python programming tasks from natural-language prompts.
- `BugFix`: repair-oriented tasks where the model must return corrected code.
- `Summary`: technical handoff summaries that preserve coding-critical facts.
- `InfoPres`: exact fact preservation for settings, limits, statuses, and values.
- `TPS`: local generation throughput in tokens per second.

Composite score weights:

```text
HumanEval 24%
MBPP      24%
BugFix   24%
Summary  16%
InfoPres 12%
```

`TPS` is reported but not included in the composite score.

## Requirements

- macOS or Linux with Python 3.9+
- LM Studio installed locally
- LM Studio `lms` CLI available, or installed at one of the standard LM Studio
  paths
- Models already downloaded in LM Studio
- LM Studio local server available at `http://127.0.0.1:1234`

This benchmark uses only Python standard library modules.

## Usage

```bash
python gemma4_senior_bench.py --dry-run
python gemma4_senior_bench.py
python gemma4_senior_bench.py --report
```

Run only selected models by matching part of the LM Studio model key or label:

```bash
python gemma4_senior_bench.py --dry-run --models gemma-4-26b-a4b-it-mlx
python gemma4_senior_bench.py --models gemma-4-26b-a4b-it-mlx
```

Use a non-default LM Studio host or port:

```bash
python gemma4_senior_bench.py --host 127.0.0.1 --port 1234
```

## Tuning Context Settings

Use `lmstudio_tuning_bench.py` when local models hang, timeout, or become slow
after loading multiple LLMs together. It is an operational benchmark rather than
a code-quality benchmark.

For each context length, it:

- unloads all currently loaded models
- loads both selected models with the same settings
- keeps both models resident in LM Studio
- tests one model at a time with workflow-shaped prompts
- records load time, response time, TPS, validity, and errors

The tuning benchmark intentionally does not send requests to both models at the
same time. It tests the workflow we expect to use operationally: both models are
resident in memory, but requests are routed sequentially to one model at a time.

The built-in scenarios are shaped around the local workflow we have been tuning:
Codex handoff summaries, BMAD orchestration plans, coding-worker patches,
review responses, latency probes, and long-context recall.

Dry-run the default two-role setup:

```bash
python lmstudio_tuning_bench.py --dry-run
```

Run a focused tuning sweep:

```bash
python lmstudio_tuning_bench.py \
  --models gemma-4-31b-dense-platinum,gemma-4-26b-a4b-it-mlx \
  --context-lengths 4096,8192,16384,32768
```

Run a wider context sweep, including large-context settings:

```bash
python lmstudio_tuning_bench.py \
  --models gemma-4-31b-dense-platinum,gemma-4-26b-a4b-it-mlx \
  --context-lengths 4096,8192,16384,32768,65000
```

Add your own historical failure scenarios with a JSON file:

```json
[
  {
    "name": "custom_codex_handoff",
    "prompt": "Long prompt text here",
    "required": ["Files Changed", "Verification", "Risks"],
    "max_tokens": 320
  }
]
```

```bash
python lmstudio_tuning_bench.py --scenarios-file scenarios.json
```

Results are saved under `results/tuningbench_<timestamp>/` as:

- `results.csv`
- `results.json`
- `summary.json`
- per-combination `lms` logs

For operational use, prefer the smallest context length that loads both models
and passes the workflow-shaped scenarios without timeouts. Larger context values
are useful when a task genuinely needs long-document handoff, but they can add
memory pressure without improving normal coding and orchestration work.

## Results

Results are saved under `results/coderbench_<timestamp>/`:

- `progress.json`: updated after each completed model
- `final_report.txt`: ranked final report
- `lms_*.log`: per-model LM Studio load/unload logs

Generated results and model files are intentionally ignored by git.

## Current Recommended Role Split

Based on local benchmark results from this environment:

- Coding worker: `CelesteImperia / Gemma 4 31B (Q4_K_M)`
- Orchestrator / architect: `lmstudio-community / Gemma 4 26B A4B Instruct (4bit)`
- Backup reviewer: `qwen / Qwen3 Coder Next`

These recommendations are environment-specific. Re-run the benchmark after
changing model quantization, publisher, LM Studio version, or hardware.
