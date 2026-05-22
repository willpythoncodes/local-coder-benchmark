#!/usr/bin/env python3
"""
LM Studio operational tuning benchmark.

This benchmark searches for stable LM Studio context-length settings for one or
two local models. For each context length it:

    1. Unloads all currently loaded models.
    2. Loads every requested model with that context length.
    3. Keeps all requested models resident together.
    4. Tests one model at a time with workflow-shaped prompts.
    5. Records load time, latency, throughput, response validity, and errors.
"""

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


LM_STUDIO_HOST = "127.0.0.1"
LM_STUDIO_PORT = 1234
SERVER_TIMEOUT = 120
LOAD_TIMEOUT = 900
RESULTS_DIR = Path(__file__).parent / "results"

DEFAULT_MODELS = [
    "gemma-4-31b-dense-platinum",
    "gemma-4-26b-a4b-it-mlx",
]
DEFAULT_CONTEXTS = [4096, 8192, 16384, 32768]


def parse_csv_ints(raw: str, *, name: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers")
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{name} values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return values


def parse_csv_strings(raw: str) -> List[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("model list cannot be empty")
    return values


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:48] or "model"


def lms_bin() -> str:
    bin_path = shutil.which("lms")
    if bin_path:
        return bin_path
    for fallback in (
        Path.home() / ".lmstudio" / "bin" / "lms",
        Path.home() / ".cache" / "lm-studio" / "bin" / "lms",
    ):
        if fallback.exists():
            return str(fallback)
    raise RuntimeError("Could not find `lms`. Open LM Studio and run `lms bootstrap`.")


def run_lms(args: List[str], *, timeout: Optional[int] = 60, log_path: Optional[Path] = None) -> subprocess.CompletedProcess:
    cmd = [lms_bin(), *args]
    started = datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(f"\n--- {started} lms {' '.join(args)} rc={proc.returncode} ---\n")
            if proc.stdout:
                f.write(proc.stdout)
            if proc.stderr:
                f.write("\n[stderr]\n")
                f.write(proc.stderr)
    return proc


def server_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def server_alive(base_url: str) -> bool:
    try:
        urllib.request.urlopen(f"{base_url}/v1/models", timeout=2)
        return True
    except Exception:
        return False


def ensure_server(base_url: str) -> None:
    if server_alive(base_url):
        return
    print(f"  [LM Studio] Starting server at {base_url} ...")
    run_lms(["server", "start"], timeout=60)
    deadline = time.time() + SERVER_TIMEOUT
    while time.time() < deadline:
        if server_alive(base_url):
            return
        time.sleep(2)
    raise RuntimeError(f"LM Studio server did not become reachable at {base_url}")


def loaded_model_ids(base_url: str) -> List[str]:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as resp:
            data = json.loads(resp.read())
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
    except Exception:
        return []


def load_model(model_key: str, identifier: str, context_length: int,
               gpu: str, log_path: Path) -> Dict:
    args = [
        "load",
        model_key,
        "--context-length",
        str(context_length),
        "--identifier",
        identifier,
        "-y",
    ]
    if gpu:
        args.extend(["--gpu", gpu])

    started = time.time()
    proc = run_lms(args, timeout=LOAD_TIMEOUT, log_path=log_path)
    elapsed = time.time() - started
    return {
        "model_key": model_key,
        "api_model_id": identifier,
        "load_seconds": round(elapsed, 2),
        "load_ok": proc.returncode == 0,
        "load_error": "" if proc.returncode == 0 else (proc.stderr or proc.stdout or "").strip()[:1000],
    }


def unload_all(log_path: Optional[Path] = None) -> None:
    run_lms(["unload", "--all"], timeout=180, log_path=log_path)
    time.sleep(1)


def approx_context_prompt(context_length: int, stress_ratio: float, max_prompt_words: int) -> str:
    target_words = max(128, int(context_length * stress_ratio))
    target_words = min(target_words, max_prompt_words)
    facts = []
    for i in range(target_words):
        facts.append(f"fact_{i:05d}=value_{(i * 17) % 9973:04d}")
    anchor_a = facts[len(facts) // 4]
    anchor_b = facts[len(facts) // 2]
    anchor_c = facts[-7]
    body = " ".join(facts)
    return (
        "You are testing long-context reliability. Read the following synthetic facts, "
        "then answer with exactly three lines in the form key=value.\n\n"
        f"{body}\n\n"
        "Return these exact facts only:\n"
        f"{anchor_a}\n{anchor_b}\n{anchor_c}\n"
    )


def default_scenarios(context_length: int, stress_ratio: float, max_prompt_words: int) -> List[Dict]:
    context_prompt = approx_context_prompt(context_length, stress_ratio, max_prompt_words)
    context_required = re.findall(r"fact_\d+=value_\d+", context_prompt.split("Return these exact facts only:", 1)[1])
    return [
        {
            "name": "latency_json",
            "prompt": (
                "Return exactly this JSON object and no markdown: "
                '{"status":"ok","task":"latency_probe","answer":42}'
            ),
            "max_tokens": 96,
            "required": ['"status"', '"ok"', "42"],
        },
        {
            "name": "codex_handoff_summary",
            "prompt": """You are summarizing a Codex coding session for a local orchestrator.

Conversation facts:
- User asked to stop an old benchmark after HumanEval/Summary/InfoPres problems.
- The active project path is /Users/willgomez/Documents/New project/draftbench-main.
- The successful benchmark backend is LM Studio at http://127.0.0.1:1234/v1.
- The coding worker is CelesteImperia / Gemma 4 31B (Q4_K_M), model key gemma-4-31b-dense-platinum.
- The orchestrator is lmstudio-community / Gemma 4 26B A4B Instruct (4bit), model key gemma-4-26b-a4b-it-mlx.
- The code must continue using BMAD workflow discipline.
- Do not include old Ollama assumptions.
- Verification previously used py_compile and dry-run.

Write a concise handoff with exactly these headings:
Goal
Decisions
Files Changed
Verification
Risks
""",
            "max_tokens": 320,
            "required": ["Goal", "Decisions", "Files Changed", "Verification", "Risks", "LM Studio", "BMAD"],
        },
        {
            "name": "bmad_orchestrator_plan",
            "prompt": """You are the principal orchestrator for a BMAD-style coding workflow.

Task: Add a small feature to an existing benchmark without overbuilding.
Constraints:
- Inspect the current code first.
- Keep the change limited to one script and README docs.
- Do not alter generated results.
- Ask the coding worker to implement only the approved plan.
- Verification must include py_compile and a dry-run command.
- Final response must summarize changed files, checks run, and risks.

Return a compact plan with exactly these headings:
Scope
Implementation Plan
Delegation To Coding Worker
Verification
Stop Conditions
""",
            "max_tokens": 360,
            "required": ["Scope", "Implementation Plan", "Delegation To Coding Worker", "Verification", "Stop Conditions"],
        },
        {
            "name": "coding_worker_patch",
            "prompt": """You are the senior coding worker. Return only Python code, no markdown.

Implement this function:

def summarize_failures(rows):
    '''
    rows is a list of dicts with keys:
    - model_key
    - test
    - ok
    - seconds
    - error

    Return a dict with:
    - total_rows
    - failed_rows
    - slowest_seconds
    - failed_models, sorted unique model_key values for rows where ok is false
    '''
""",
            "max_tokens": 320,
            "required": ["def summarize_failures", "total_rows", "failed_rows", "slowest_seconds", "failed_models"],
        },
        {
            "name": "review_findings",
            "prompt": """Review this proposed benchmark behavior:

The script loads two local LM Studio models, runs long context prompts, and records latency. It writes JSON and CSV after every context-length test. It ignores generated results in git. It does not run the two models at the same time, but keeps both loaded.

Return a concise code-review style response with exactly these headings:
Findings
Missing Tests
Operational Risks
Recommended Next Step
""",
            "max_tokens": 320,
            "required": ["Findings", "Missing Tests", "Operational Risks", "Recommended Next Step"],
        },
        {
            "name": "long_context_recall",
            "prompt": context_prompt,
            "max_tokens": 160,
            "required": context_required,
        },
    ]


def load_scenarios(path: Optional[str], context_length: int,
                   stress_ratio: float, max_prompt_words: int) -> List[Dict]:
    scenarios = default_scenarios(context_length, stress_ratio, max_prompt_words)
    if not path:
        return scenarios

    with open(path) as f:
        custom = json.load(f)
    if not isinstance(custom, list):
        raise ValueError("--scenarios-file must contain a JSON list of scenario objects")

    loaded = []
    for i, item in enumerate(custom, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario {i} is not an object")
        name = str(item.get("name") or f"custom_{i}")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Scenario {name!r} is missing a non-empty prompt")
        required = item.get("required", [])
        if not isinstance(required, list) or not all(isinstance(v, str) for v in required):
            raise ValueError(f"Scenario {name!r} required must be a list of strings")
        loaded.append({
            "name": name,
            "prompt": prompt,
            "max_tokens": int(item.get("max_tokens") or 320),
            "required": required,
        })
    return scenarios + loaded


def call_model(base_url: str, model_id: str, prompt: str, *,
               max_tokens: int, timeout: int) -> Dict:
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - started
        content = data["choices"][0]["message"].get("content", "")
        tokens = data.get("usage", {}).get("completion_tokens") or max(1, len(content.split()))
        return {
            "ok": True,
            "seconds": round(elapsed, 3),
            "tokens": tokens,
            "tps": round(tokens / elapsed, 2) if elapsed > 0 else 0.0,
            "chars": len(content),
            "preview": content[:240].replace("\n", "\\n"),
            "error": "",
        }
    except urllib.error.URLError as e:
        return {
            "ok": False,
            "seconds": round(time.time() - started, 3),
            "tokens": 0,
            "tps": 0.0,
            "chars": 0,
            "preview": "",
            "error": str(e)[:1000],
        }
    except Exception as e:
        return {
            "ok": False,
            "seconds": round(time.time() - started, 3),
            "tokens": 0,
            "tps": 0.0,
            "chars": 0,
            "preview": "",
            "error": str(e)[:1000],
        }


def response_valid(scenario: Dict, result: Dict) -> bool:
    if not result["ok"]:
        return False
    preview = result.get("preview", "")
    required = scenario.get("required") or []
    if required:
        haystack = preview.lower()
        return all(needle.lower() in haystack for needle in required)
    return result["chars"] > 0


def summarize_combo(rows: List[Dict]) -> Dict:
    test_rows = [r for r in rows if r["phase"] == "test"]
    load_rows = [r for r in rows if r["phase"] == "load"]
    failures = [r for r in rows if not r["ok"]]
    test_seconds = [r["seconds"] for r in test_rows if isinstance(r.get("seconds"), (int, float))]
    tps_values = [r["tps"] for r in test_rows if isinstance(r.get("tps"), (int, float)) and r["tps"] > 0]
    return {
        "load_ok": all(r["ok"] for r in load_rows),
        "tests_ok": all(r["ok"] for r in test_rows),
        "valid_ok": all(r.get("valid", False) for r in test_rows),
        "failure_count": len(failures),
        "max_test_seconds": max(test_seconds) if test_seconds else math.inf,
        "avg_tps": round(sum(tps_values) / len(tps_values), 2) if tps_values else 0.0,
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    fieldnames = [
        "run_id", "context_length", "model_key", "api_model_id", "phase",
        "scenario", "ok", "valid", "seconds", "tps", "tokens", "chars",
        "load_seconds", "required", "error", "preview",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune LM Studio context length with workflow-shaped scenarios.")
    parser.add_argument("--models", type=parse_csv_strings, default=DEFAULT_MODELS,
                        help="Comma-separated LM Studio model keys to load together.")
    parser.add_argument("--context-lengths", type=lambda v: parse_csv_ints(v, name="context-lengths"),
                        default=DEFAULT_CONTEXTS,
                        help="Comma-separated context lengths to test.")
    parser.add_argument("--host", default=LM_STUDIO_HOST)
    parser.add_argument("--port", type=int, default=LM_STUDIO_PORT)
    parser.add_argument("--gpu", default="", help='Optional LM Studio --gpu value, e.g. "max", "off", or "0.5".')
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=384,
                        help="Fallback max_tokens for custom scenarios that do not specify max_tokens.")
    parser.add_argument("--stress-ratio", type=float, default=0.35,
                        help="Approximate prompt size as a fraction of context length.")
    parser.add_argument("--max-prompt-words", type=int, default=12000,
                        help="Safety cap for synthetic context prompt size.")
    parser.add_argument("--scenarios-file", default="",
                        help="Optional JSON list of additional scenarios with name, prompt, required, max_tokens.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.stress_ratio <= 0 or args.stress_ratio > 1:
        raise SystemExit("--stress-ratio must be greater than 0 and less than or equal to 1")

    base_url = server_url(args.host, args.port)
    dry_run_scenarios = load_scenarios(
        args.scenarios_file or None,
        args.context_lengths[0],
        args.stress_ratio,
        args.max_prompt_words,
    )

    print("LM Studio tuning benchmark")
    print(f"Endpoint: {base_url}")
    print(f"Models:   {', '.join(args.models)}")
    print(f"Contexts: {', '.join(map(str, args.context_lengths))}")
    print(f"Scenarios: {len(dry_run_scenarios)} workflow-shaped prompt(s)")
    print(f"Combos:   {len(args.context_lengths)}")

    if args.dry_run:
        for scenario in dry_run_scenarios:
            required = scenario.get("required") or []
            print(f"  - {scenario['name']} ({len(required)} required marker(s))")
        return

    ensure_server(base_url)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("tuningbench_%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict] = []
    summary_rows: List[Dict] = []

    try:
        for combo_index, context_length in enumerate(args.context_lengths, 1):
            print(f"\n[{combo_index}/{len(args.context_lengths)}] context={context_length}")
            log_path = run_dir / f"ctx{context_length}.log"
            unload_all(log_path)
            scenarios = load_scenarios(
                args.scenarios_file or None,
                context_length,
                args.stress_ratio,
                args.max_prompt_words,
            )

            combo_rows: List[Dict] = []
            identifiers: Dict[str, str] = {}
            load_failed = False
            for model_index, model_key in enumerate(args.models, 1):
                identifier = f"tune_{model_index}_{slug(model_key)}_ctx{context_length}"
                identifiers[model_key] = identifier
                print(f"  Loading {model_key} as {identifier}")
                load = load_model(model_key, identifier, context_length, args.gpu, log_path)
                row = {
                    "run_id": run_id,
                    "context_length": context_length,
                    "model_key": model_key,
                    "api_model_id": identifier,
                    "phase": "load",
                    "scenario": "load",
                    "ok": load["load_ok"],
                    "valid": load["load_ok"],
                    "seconds": load["load_seconds"],
                    "tps": "",
                    "tokens": "",
                    "chars": "",
                    "load_seconds": load["load_seconds"],
                    "required": "",
                    "error": load["load_error"],
                    "preview": "",
                }
                combo_rows.append(row)
                all_rows.append(row)
                if not load["load_ok"]:
                    load_failed = True
                    print(f"    LOAD FAILED: {load['load_error'][:160]}")

            print(f"  Loaded API ids: {', '.join(loaded_model_ids(base_url))}")

            if not load_failed:
                for model_key in args.models:
                    model_id = identifiers[model_key]
                    for scenario in scenarios:
                        scenario_name = scenario["name"]
                        max_tokens = int(scenario.get("max_tokens") or args.max_tokens)
                        print(f"  Testing {model_key} [{scenario_name}]")
                        result = call_model(
                            base_url,
                            model_id,
                            scenario["prompt"],
                            max_tokens=max_tokens,
                            timeout=args.request_timeout,
                        )
                        valid = response_valid(scenario, result)
                        row = {
                            "run_id": run_id,
                            "context_length": context_length,
                            "model_key": model_key,
                            "api_model_id": model_id,
                            "phase": "test",
                            "scenario": scenario_name,
                            "ok": result["ok"],
                            "valid": valid,
                            "seconds": result["seconds"],
                            "tps": result["tps"],
                            "tokens": result["tokens"],
                            "chars": result["chars"],
                            "load_seconds": "",
                            "required": "; ".join(scenario.get("required") or []),
                            "error": result["error"],
                            "preview": result["preview"],
                        }
                        combo_rows.append(row)
                        all_rows.append(row)
                        status = "ok" if row["ok"] and row["valid"] else "fail"
                        print(f"    {status}: {row['seconds']}s, {row['tps']} tok/s")

            combo_summary = summarize_combo(combo_rows)
            combo_summary.update({
                "run_id": run_id,
                "context_length": context_length,
            })
            summary_rows.append(combo_summary)

            write_csv(run_dir / "results.csv", all_rows)
            with open(run_dir / "results.json", "w") as f:
                json.dump(all_rows, f, indent=2)
            with open(run_dir / "summary.json", "w") as f:
                json.dump(summary_rows, f, indent=2)

        successful = [
            row for row in summary_rows
            if row["load_ok"] and row["tests_ok"] and row["valid_ok"]
        ]
        if successful:
            best = sorted(successful, key=lambda r: (r["max_test_seconds"], -r["avg_tps"]))[0]
            print(
                "\nRecommended stable setting: "
                f"context={best['context_length']} "
                f"(max response {best['max_test_seconds']:.3f}s, avg TPS {best['avg_tps']})"
            )
        else:
            print("\nNo setting completed every load/test/validity check successfully.")
        print(f"Results saved: {run_dir}")
    finally:
        unload_all()


if __name__ == "__main__":
    main()
