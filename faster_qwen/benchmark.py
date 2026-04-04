"""
benchmark.py - TTS performance benchmark across all flag combinations.

Tests every combination of INT8_QUANTIZE × HYBRID_MODE × KV_CACHE_INT8 (8 combos).
Each combo loads the model fresh, runs warmup, then synthesizes all test sentences
3 times and reports timing + VRAM stats.

Usage:
    python benchmark.py [--ref path/to/ref.wav] [--runs N] [--model MODEL_NAME]

Output: printed table + benchmark_results.json
"""
import argparse
import gc
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANCHOR_TEXT = "안녕하세요, 읽어주는 별코코입니다. 오늘도 좋은 하루 되세요."

TEST_SENTENCES = [
    "안녕하세요, 오늘 날씨가 정말 좋네요.",
    "빠른 갈색 여우가 게으른 개를 뛰어넘었습니다.",
    "이 문장은 조금 더 길게 읽어보는 테스트입니다. 잘 들리시나요?",
    "하나, 둘, 셋, 넷, 다섯, 여섯, 일곱, 여덟, 아홉, 열.",
    "저는 오늘 커피 두 잔을 마셨는데도 여전히 졸립니다.",
]

DEFAULT_MODEL  = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_REF    = str(Path(__file__).parent / "references" / "default_server.wav")
DEFAULT_RUNS   = 3


# ── VRAM helpers ─────────────────────────────────────────────────────────────

def vram_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2

def vram_reserved_mb() -> float:
    return torch.cuda.memory_reserved() / 1024**2

def peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


# ── Model loader (mirrors bot.py _load_gpu_engine) ───────────────────────────

def load_model(model_name: str, int8_quantize: bool, hybrid_mode: bool, kv_cache_int8: bool):
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"  Loading model…")
    model = FasterQwen3TTS.from_pretrained(model_name, device="cuda", dtype=torch.bfloat16)

    if int8_quantize:
        try:
            from qwen3_tts_triton.models.patching import find_patchable_model
            from torchao.quantization import Int8WeightOnlyConfig, quantize_
            internal = find_patchable_model(model.model)
            quantize_(internal, Int8WeightOnlyConfig())
            print("  [ok] Int8 weight-only quantization")
        except Exception as e:
            print(f"  [skip] Int8 quantize: {e}")

    if hybrid_mode:
        try:
            from qwen3_tts_triton.models.patching import find_patchable_model, apply_triton_kernels
            internal = find_patchable_model(model.model)
            apply_triton_kernels(internal)
            print("  [ok] Triton hybrid kernels")
        except Exception as e:
            print(f"  [skip] Triton: {e}")

    if int8_quantize:
        try:
            model.predictor_graph.pred_model = torch.compile(
                model.predictor_graph.pred_model, mode="max-autotune-no-cudagraphs")
            model.talker_graph.model = torch.compile(
                model.talker_graph.model, mode="max-autotune-no-cudagraphs")
            print("  [ok] torch.compile")
        except Exception as e:
            print(f"  [skip] torch.compile: {e}")

    if kv_cache_int8:
        try:
            from kv_cache_quant import replace_static_caches
            n = replace_static_caches(model)
            print(f"  [ok] KV cache int8 ({n} layers)")
        except Exception as e:
            print(f"  [skip] KV cache int8: {e}")

    return model


# ── Warmup ────────────────────────────────────────────────────────────────────

def warmup(model, ref_path: str):
    print("  Warming up (CUDA graph capture)…")
    t0 = time.monotonic()
    for chunk in model.generate_voice_clone_streaming(
        text=ANCHOR_TEXT,
        language="Korean",
        ref_audio=ref_path,
        ref_text=ANCHOR_TEXT,
        chunk_size=4,
        xvec_only=False,
    ):
        pass
    elapsed = time.monotonic() - t0
    torch.cuda.synchronize()
    print(f"  Warmup done in {elapsed:.1f}s")


# ── Single synthesis run ──────────────────────────────────────────────────────

def synthesize_one(model, text: str, ref_path: str) -> dict:
    """Returns timing + audio stats for one synthesis."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t_start = time.monotonic()

    chunks = []
    first_chunk_t = None
    for audio_chunk, sr, timing in model.generate_voice_clone_streaming(
        text=text,
        language="Korean",
        ref_audio=ref_path,
        ref_text=ANCHOR_TEXT,
        chunk_size=4,
        xvec_only=False,
    ):
        if first_chunk_t is None:
            first_chunk_t = time.monotonic()
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.float().cpu().numpy()
        chunks.append(audio_chunk)

    torch.cuda.synchronize()
    t_end = time.monotonic()

    audio = np.concatenate(chunks) if chunks else np.array([])
    audio_duration = len(audio) / sr if chunks else 0.0
    total_time = t_end - t_start
    ttfb = (first_chunk_t - t_start) if first_chunk_t else total_time

    return {
        "total_ms":     total_time * 1000,
        "ttfb_ms":      ttfb * 1000,
        "audio_dur_s":  audio_duration,
        "rtf":          total_time / audio_duration if audio_duration > 0 else float("inf"),
        "peak_vram_mb": peak_vram_mb(),
    }


# ── Run one combo ─────────────────────────────────────────────────────────────

def run_combo(combo: dict, model_name: str, ref_path: str, runs: int) -> dict:
    label = (
        f"INT8={'Y' if combo['int8_quantize'] else 'N'}  "
        f"HYBRID={'Y' if combo['hybrid_mode'] else 'N'}  "
        f"KV_INT8={'Y' if combo['kv_cache_int8'] else 'N'}"
    )
    print(f"\n{'─'*60}")
    print(f"  Combo: {label}")
    print(f"{'─'*60}")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model = load_model(
        model_name,
        int8_quantize=combo["int8_quantize"],
        hybrid_mode=combo["hybrid_mode"],
        kv_cache_int8=combo["kv_cache_int8"],
    )

    vram_after_load = vram_mb()
    warmup(model, ref_path)
    vram_after_warmup = vram_mb()

    # Per-sentence results across all runs
    sentence_results: list[list[dict]] = [[] for _ in TEST_SENTENCES]

    for run_idx in range(runs):
        print(f"  Run {run_idx + 1}/{runs}…")
        for sent_idx, sentence in enumerate(TEST_SENTENCES):
            r = synthesize_one(model, sentence, ref_path)
            sentence_results[sent_idx].append(r)

    # Aggregate
    sentences_agg = []
    for sent_idx, sentence in enumerate(TEST_SENTENCES):
        rlist = sentence_results[sent_idx]
        agg = {
            "text":           sentence,
            "avg_total_ms":   np.mean([r["total_ms"]    for r in rlist]),
            "avg_ttfb_ms":    np.mean([r["ttfb_ms"]     for r in rlist]),
            "avg_audio_dur_s":np.mean([r["audio_dur_s"] for r in rlist]),
            "avg_rtf":        np.mean([r["rtf"]          for r in rlist]),
            "avg_peak_vram":  np.mean([r["peak_vram_mb"] for r in rlist]),
        }
        sentences_agg.append(agg)

    overall_rtf   = np.mean([s["avg_rtf"]       for s in sentences_agg])
    overall_ttfb  = np.mean([s["avg_ttfb_ms"]   for s in sentences_agg])
    overall_total = np.mean([s["avg_total_ms"]  for s in sentences_agg])

    print(f"\n  Results (avg over {runs} runs):")
    print(f"  {'Sentence':<55} {'RTF':>6} {'TTFB ms':>9} {'Total ms':>10}")
    print(f"  {'─'*55} {'─'*6} {'─'*9} {'─'*10}")
    for s in sentences_agg:
        preview = s["text"][:52] + "…" if len(s["text"]) > 52 else s["text"]
        print(f"  {preview:<55} {s['avg_rtf']:>6.3f} {s['avg_ttfb_ms']:>9.1f} {s['avg_total_ms']:>10.1f}")
    print(f"  {'─'*55} {'─'*6} {'─'*9} {'─'*10}")
    print(f"  {'OVERALL':<55} {overall_rtf:>6.3f} {overall_ttfb:>9.1f} {overall_total:>10.1f}")
    print(f"  VRAM after load: {vram_after_load:.1f} MB   after warmup: {vram_after_warmup:.1f} MB")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label":            label,
        "flags":            combo,
        "vram_after_load_mb":   vram_after_load,
        "vram_after_warmup_mb": vram_after_warmup,
        "overall_rtf":      overall_rtf,
        "overall_ttfb_ms":  overall_ttfb,
        "overall_total_ms": overall_total,
        "sentences":        sentences_agg,
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    print(f"\n{'═'*85}")
    print("  SUMMARY")
    print(f"{'═'*85}")
    print(f"  {'Flags (INT8 / HYBRID / KV)':<28} {'RTF':>6} {'TTFB ms':>9} {'Total ms':>10} {'VRAM load':>11} {'VRAM warm':>11}")
    print(f"  {'─'*28} {'─'*6} {'─'*9} {'─'*10} {'─'*11} {'─'*11}")
    for r in results:
        flags = r["flags"]
        label = f"{'Y' if flags['int8_quantize'] else 'N'} / {'Y' if flags['hybrid_mode'] else 'N'} / {'Y' if flags['kv_cache_int8'] else 'N'}"
        print(
            f"  {label:<28} {r['overall_rtf']:>6.3f} {r['overall_ttfb_ms']:>9.1f}"
            f" {r['overall_total_ms']:>10.1f} {r['vram_after_load_mb']:>9.1f} MB"
            f" {r['vram_after_warmup_mb']:>9.1f} MB"
        )
    print(f"{'═'*85}")
    print("  RTF < 1.0 = faster than real-time  |  lower is better for all metrics")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TTS benchmark across flag combinations")
    parser.add_argument("--ref",   default=DEFAULT_REF,   help="Reference WAV path for voice cloning")
    parser.add_argument("--runs",  type=int, default=DEFAULT_RUNS, help="Synthesis runs per sentence per combo")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name or local path")
    parser.add_argument("--combos", nargs="*",
                        help="Subset of combos to run, e.g. '0 3 7' (0-7). Default: all 8.")
    args = parser.parse_args()

    if not Path(args.ref).exists():
        print(f"ERROR: reference WAV not found: {args.ref}")
        print("Pass --ref path/to/voice.wav")
        sys.exit(1)

    # All 8 combinations of (int8_quantize, hybrid_mode, kv_cache_int8)
    all_combos = [
        {"int8_quantize": i, "hybrid_mode": h, "kv_cache_int8": k}
        for i, h, k in product([False, True], repeat=3)
    ]

    selected = all_combos
    if args.combos:
        indices = [int(x) for x in args.combos]
        selected = [all_combos[i] for i in indices]

    print(f"Benchmarking {len(selected)} combo(s), {args.runs} run(s) each, {len(TEST_SENTENCES)} sentences")
    print(f"Model : {args.model}")
    print(f"Ref   : {args.ref}")

    all_results = []
    for combo in selected:
        result = run_combo(combo, args.model, args.ref, args.runs)
        all_results.append(result)

    print_summary(all_results)

    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
