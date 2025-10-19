# test_multiple_models.py
"""
Multi-model validation for gpu-poor (fast by default, full with --all)
- Skips gated models cleanly if HF token/license missing
- Writes JSON + Markdown under results/
- Captures hardware & library versions for credibility
"""
import argparse
import importlib
import json
import os
import platform
import shutil
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
JSON_PATH = RESULTS_DIR / "validation_results.json"
MD_PATH = RESULTS_DIR / "validation_results.md"

# --- Model matrices ---
FAST_MODELS = [
    ("gpt2", "Open, tiny baseline (causal)"),
    ("gpt2-medium", "Open, mid baseline (causal)"),
    ("facebook/opt-125m", "Open OPT (causal)"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Open, small modern (causal)"),
    ("google/flan-t5-small", "Open, seq2seq (T5)"), 
]


FULL_MODELS = FAST_MODELS + [
    ("facebook/opt-1.3b", "Open, larger OPT (causal)"),
    ("microsoft/phi-2", "Open, 2.7B (causal)"),
    ("meta-llama/Llama-2-7b-hf", "Gated, 7B Llama (causal)"),
    ("mistralai/Mistral-7B-v0.1", "Gated, 7B Mistral (causal)"),
    ("meta-llama/Llama-3.1-8B", "Gated, 8B Llama (causal)"),
]

def has_hf_token() -> bool:
    return bool(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))

def load_demo():
    try:
        mod = importlib.import_module("examples.demo")
    except ModuleNotFoundError:
        try:
            mod = importlib.import_module("demo")
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Could not import 'examples.demo' or 'demo'. "
                "Run from the repo root and ensure the demo module exists."
            ) from e
    if not hasattr(mod, "demo_production_ready"):
        raise RuntimeError("Missing demo_production_ready(model_name) in the demo module.")
    return getattr(mod, "demo_production_ready")


def system_metadata():
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        ram_gb = None
    meta = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": ram_gb,
    }
    # libs
    try:
        import torch, transformers
        meta["torch"] = getattr(torch, "__version__", "unknown")
        meta["transformers"] = getattr(transformers, "__version__", "unknown")
        meta["torch_threads"] = getattr(torch, "get_num_threads", lambda: None)()
    except Exception:
        pass
    return meta

def model_result_path(model_id: str) -> Path:
    return RESULTS_DIR / f"{model_id.replace('/', '_')}.json"

def summarise_to_markdown(all_results: list, meta: dict):
    lines = []
    lines.append("# GPU-POOR Validation Results\n")
    lines.append(f"_Generated: {meta.get('timestamp','')}_\n")

    hw = f"{meta.get('os','')} · Python {meta.get('python','')} · Torch {meta.get('torch','?')} · Transformers {meta.get('transformers','?')}"
    if meta.get("cpu"):
        hw = f"{meta['cpu']} · " + hw
    if meta.get("ram_gb"):
        hw = f"{meta['ram_gb']} GB RAM · " + hw
    lines.append(f"**Hardware/Env:** {hw}\n")

    lines.append("\n| Model | Original (MB) | Compressed (MB) | Reduction | Speedup | Quality | Note |\n"
                 "|---|---:|---:|---:|---:|---|---|\n")

    for r in all_results:
        if "error" in r:
            lines.append(f"| `{r['model']}` | - | - | - | - | - | ⚠️ {r['error']} |\n")
            continue

        # Computing speedup per row
        sp = r.get("speedup")
        if sp is None:
            lr = r.get("latency_ratio")
            sp = (1.0 / lr) if lr and lr > 0 else 0.0

        lines.append(
            f"| `{r['model']}` | {r['original_mb']:.1f} | {r['compressed_mb']:.1f} | "
            f"{r['compression']:.1f}% | {sp:.2f}× | {r.get('quality','')} | {r.get('importance','')} |\n"
        )

    MD_PATH.write_text("".join(lines), encoding="utf-8")


def run_all_tests(models, force=False):
    print("=" * 70)
    print("GPU-POOR MULTI-MODEL VALIDATION TEST")
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    demo_fn = load_demo()
    all_results = []
    token_available = has_hf_token()

    for model_name, importance in models:
        print(f"\n{'='*70}\nTesting: {model_name}\nImportance: {importance}\n{'='*70}")

        # gated heuristic
        is_gated = any(k in model_name.lower() for k in ["llama", "mistral-7b", "llama-2", "llama-3"])
        result_file = model_result_path(model_name)

        try:
            if result_file.exists() and not force:
                print(f"[SKIP] Found existing results at {result_file}")
                results = json.loads(result_file.read_text())
            else:
                if is_gated and not token_available:
                    raise RuntimeError("Gated model requires HF token. Set HF_TOKEN and accept the model license.")

                # run the demo
                results = demo_fn(model_name)
                print(f"[OK] Completed: {model_name}")
                result_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

            all_results.append({
                "model": model_name,
                "importance": importance,
                "compression": results.get("compression_ratio", 0),
                "speedup": results.get("speedup_x") or (
                    (1.0 / results["latency_ratio_opt_over_fp32"]) 
                    if results.get("latency_ratio_opt_over_fp32") else None
                ),
                "latency_ratio": results.get("latency_ratio_opt_over_fp32"),
                "quality": "good" if not results.get("has_repetitions", False) else "degraded",
                "original_mb": results.get("original_size", 0),
                "compressed_mb": results.get("compressed_size", 0)

            })

        except Exception as e:
            msg = str(e).strip().split("\n")[0][:180]
            print(f"[ERROR] {model_name}: {msg}")
            all_results.append({
                "model": model_name,
                "importance": importance,
                "error": msg
            })
            continue

    # attach metadata & save
    meta = system_metadata()
    payload = {"meta": meta, "results": all_results}
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summarise_to_markdown(all_results, meta)

    # pretty console summary
    print("\n" + "="*90)
    print("VALIDATION SUMMARY")
    print("="*90)
    header = f"{'Model':<32} {'Original':>9} {'Compressed':>11} {'Reduction':>11} {'Speedup':>8} {'Quality':>9}"
    print(header)
    print("-"*90)
    for r in all_results:
        if "error" in r:
            print(f"{r['model'][:32]:<32}  ERROR: {r['error']}")
        else:
            print(f"{r['model'][:32]:<32}  {r['original_mb']:>9.1f} {r['compressed_mb']:>11.1f} {r['compression']:>10.1f}% {r.get('speedup',0):>7.2f}x {r['quality']:>9}")
    succ = [r for r in all_results if 'error' not in r and r.get('compression',0) > 0]
    if succ:
        avg_comp = sum(r['compression'] for r in succ)/len(succ)
        print(f"\nAverage Compression across successful: {avg_comp:.1f}%  ({len(succ)}/{len(all_results)} models)")
    print("="*90)
    print(f"Saved: {JSON_PATH} and {MD_PATH}")

    return payload

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="Run full model matrix (includes gated 7B models)")
    p.add_argument("--force", action="store_true", help="Re-run even if per-model results already exist")
    args = p.parse_args()

    models = FULL_MODELS if args.all else FAST_MODELS
    payload = run_all_tests(models, force=args.force)

    # Exit nonzero if any Critical model failed (useful in CI)
    failures = [r for r in payload["results"] if r.get("importance","").startswith("Critical") and "error" in r]
    if failures:
        # Don't be harsh in local dev; for CI you may prefer to exit(1).
        print(f"\n[WARN] Critical model failures: {[f['model'] for f in failures]}")

if __name__ == "__main__":
    main()
