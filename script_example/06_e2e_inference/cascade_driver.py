import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

@dataclass
class RequestMetrics:
    request_id: int
    session_id: int
    turn: int
    cache_hit: bool

    prefill_time_ms: float
    cascade_put_time_ms: float
    cascade_get_time_ms: float
    serialize_time_ms: float
    deserialize_time_ms: float
    decode_time_ms: float
    e2e_ttft_ms: float

    num_prefix_tokens: int
    num_new_tokens: int
    num_generated_tokens: int
    kv_cache_bytes: int

def run_benchmark(args):
    import torch
    from .config import ModelConfig, CascadeConfig, BenchmarkConfig, QWEN_72B, LLAMA_70B
    from .engine import InferenceEngine
    from .kv_manager import KVCacheStore, compute_prefix_hash
    from .workload import ShareGPTWorkload
    from benchmark.run_benchmark import get_adapter

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    model_presets = {
        "Qwen/Qwen2.5-72B": QWEN_72B,
        "meta-llama/Llama-3-70B": LLAMA_70B,
    }
    model_cfg = model_presets.get(args.model, QWEN_72B)
    model_cfg.name = args.model
    if args.tp_size:
        model_cfg.tp_size = args.tp_size
    model_cfg.quantization = args.quantization

    cascade_cfg = {
        "gpu_capacity_gb": args.gpu_capacity,
        "shm_capacity_gb": args.shm_capacity,
        "use_gpu": True,
        "use_compression": not args.no_compression,
        "lustre_path": args.lustre_path,
    }

    log("=" * 70)
    log(f"Inference Benchmark: {model_cfg.name}")
    log(f"  KV block size: {model_cfg.kv_block_bytes / 1024 / 1024:.0f} MB")
    log(f"  TP: {model_cfg.tp_size}, Sessions: {args.num_sessions}")
    log(f"  Rank: {rank}/{world}")
    log("=" * 70)

    log("\n[1/4] Loading model...")
    engine = InferenceEngine(model_cfg)
    engine.load()

    log("\n[2/4] Initializing Cascade...")
    adapter = get_adapter("cascade", cascade_cfg)
    if not adapter.initialize():
        log("ERROR: Cascade initialization failed")
        return
    kv_store = KVCacheStore(adapter, model_cfg)

    log("\n[3/4] Generating workload...")
    workload = ShareGPTWorkload(args.dataset, tokenizer=engine.tokenizer)
    if args.prefix_tokens > 0:
        workload.set_prefix_length(args.prefix_tokens, tokenizer=engine.tokenizer)
    workload.load(max_conversations=args.num_sessions * 2)
    requests = workload.generate_requests(
        num_sessions=args.num_sessions,
        turns_per_session=args.turns_per_session,
    )

    my_requests = requests[rank::world]
    log(f"  This rank: {len(my_requests)} requests")

    log("\n[4/4] Running inference...")
    metrics: List[RequestMetrics] = []
    device = next(engine.model.parameters()).device

    if args.warmup > 0:
        log(f"  Warmup: {args.warmup} requests...")
        for req in my_requests[:args.warmup]:
            input_ids = engine.tokenize(req.full_prompt)
            with torch.no_grad():
                _ = engine.model(input_ids[:, :32], use_cache=False)
        torch.cuda.synchronize()

    for i, req in enumerate(my_requests):
        t_start = time.perf_counter()

        input_ids = engine.tokenize(req.full_prompt)
        prefix_ids = engine.tokenize(req.prefix_prompt)
        prefix_hash = compute_prefix_hash(prefix_ids, prefix_ids.shape[1])

        cached_kv = kv_store.load(prefix_hash, device=device)

        if cached_kv is not None:

            cache_hit = True
            prefill_ms = 0.0
            put_ms = 0.0
            ser_ms = 0.0

            from transformers import DynamicCache
            cache = DynamicCache()
            device_map = engine.model.hf_device_map

            model_dtype = next(p for p in engine.model.parameters() if p.is_floating_point()).dtype
            for layer_idx, (k, v) in enumerate(cached_kv):
                layer_name = f"model.layers.{layer_idx}"
                layer_device = torch.device(f"cuda:{device_map.get(layer_name, 0)}")
                cache.update(
                    k.to(dtype=model_dtype, device=layer_device),
                    v.to(dtype=model_dtype, device=layer_device),
                    layer_idx,
                )

            new_ids = input_ids[:, prefix_ids.shape[1]:]
            if new_ids.shape[1] > 0:
                torch.cuda.synchronize()
                t_pf = time.perf_counter()
                with torch.no_grad():
                    outputs = engine.model(
                        input_ids=new_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
                torch.cuda.synchronize()
                prefill_ms = (time.perf_counter() - t_pf) * 1000
                past_kv = outputs.past_key_values
            else:
                past_kv = cache

            get_ms = kv_store.stats["get_time_ms"][-1]
            deser_ms = kv_store.stats["deserialize_time_ms"][-1] if kv_store.stats["deserialize_time_ms"] else 0

        else:

            cache_hit = False
            get_ms = kv_store.stats["get_time_ms"][-1] if kv_store.stats["get_time_ms"] else 0
            deser_ms = 0.0

            prefill_result = engine.prefill(input_ids)
            prefill_ms = prefill_result.prefill_time_ms
            past_kv = prefill_result.past_key_values

            prefix_len = prefix_ids.shape[1]
            from .kv_manager import extract_kv_pairs
            kv_pairs = extract_kv_pairs(past_kv)
            if i == 0:
                log(f"  [DEBUG] past_kv type: {type(past_kv)}")
                log(f"  [DEBUG] kv_pairs len: {len(kv_pairs)}, first elem type: {type(kv_pairs[0])}")
                if hasattr(kv_pairs[0], '__len__'):
                    log(f"  [DEBUG] first elem len: {len(kv_pairs[0])}")
            prefix_kv = tuple(
                (k[:, :, :prefix_len, :], v[:, :, :prefix_len, :])
                for k, v in kv_pairs
            )
            kv_store.store(prefix_hash, prefix_kv)
            put_ms = kv_store.stats["put_time_ms"][-1]
            ser_ms = kv_store.stats["serialize_time_ms"][-1]

        torch.cuda.synchronize()
        t_decode = time.perf_counter()
        with torch.no_grad():
            current_ids = input_ids[:, -1:]
            outputs_first = engine.model(
                input_ids=current_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
        torch.cuda.synchronize()
        first_token_ms = (time.perf_counter() - t_decode) * 1000

        e2e_ms = (time.perf_counter() - t_start) * 1000

        decode_result = engine.decode(input_ids, outputs_first.past_key_values, max(args.max_new_tokens - 1, 0))

        m = RequestMetrics(
            request_id=req.request_id,
            session_id=req.session_id,
            turn=req.turn,
            cache_hit=cache_hit,
            prefill_time_ms=prefill_ms,
            cascade_put_time_ms=put_ms,
            cascade_get_time_ms=get_ms,
            serialize_time_ms=ser_ms,
            deserialize_time_ms=deser_ms,
            decode_time_ms=first_token_ms + decode_result.decode_time_ms,
            e2e_ttft_ms=e2e_ms,
            num_prefix_tokens=prefix_ids.shape[1],
            num_new_tokens=input_ids.shape[1] - prefix_ids.shape[1],
            num_generated_tokens=decode_result.num_generated_tokens,
            kv_cache_bytes=model_cfg.kv_block_bytes,
        )
        metrics.append(m)

        if i < 5 or i % 20 == 0 or i == len(my_requests) - 1:
            hit_str = "HIT " if cache_hit else "MISS"
            log(f"  [{i+1}/{len(my_requests)}] {hit_str} | "
                f"prefill={prefill_ms:.1f}ms, get={get_ms:.1f}ms, "
                f"decode={decode_result.decode_time_ms:.1f}ms, "
                f"e2e={e2e_ms:.1f}ms")

    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)

    hits = [m for m in metrics if m.cache_hit]
    misses = [m for m in metrics if not m.cache_hit]

    log(f"\nRequests: {len(metrics)} total ({len(hits)} hits, {len(misses)} misses)")
    log(f"Cache hit rate: {len(hits)/len(metrics):.1%}" if metrics else "N/A")

    if misses:
        miss_ttfts = [m.e2e_ttft_ms for m in misses]
        miss_prefills = [m.prefill_time_ms for m in misses]
        log(f"\nCACHE MISS (full prefill):")
        log(f"  E2E TTFT:  avg={np.mean(miss_ttfts):.1f}ms, "
            f"P50={np.percentile(miss_ttfts, 50):.1f}ms, "
            f"P99={np.percentile(miss_ttfts, 99):.1f}ms")
        log(f"  Prefill:   avg={np.mean(miss_prefills):.1f}ms")
        log(f"  Cascade PUT: avg={np.mean([m.cascade_put_time_ms for m in misses]):.1f}ms")

    if hits:
        hit_ttfts = [m.e2e_ttft_ms for m in hits]
        hit_gets = [m.cascade_get_time_ms for m in hits]
        log(f"\nCACHE HIT (Cascade retrieval):")
        log(f"  E2E TTFT:  avg={np.mean(hit_ttfts):.1f}ms, "
            f"P50={np.percentile(hit_ttfts, 50):.1f}ms, "
            f"P99={np.percentile(hit_ttfts, 99):.1f}ms")
        log(f"  Cascade GET: avg={np.mean(hit_gets):.1f}ms")
        log(f"  Deserialize: avg={np.mean([m.deserialize_time_ms for m in hits]):.1f}ms")

    if hits and misses:
        speedup = np.mean([m.e2e_ttft_ms for m in misses]) / np.mean([m.e2e_ttft_ms for m in hits])
        total_saved_ms = sum(m.prefill_time_ms for m in misses) * (len(hits) / max(len(misses), 1))
        log(f"\nSPEEDUP: {speedup:.1f}x (hit vs miss E2E TTFT)")
        log(f"GPU compute saved: {total_saved_ms/1000:.1f}s total")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    results = {
        "config": {
            "model": model_cfg.name,
            "kv_block_mb": model_cfg.kv_block_bytes / 1024 / 1024,
            "tp_size": model_cfg.tp_size,
            "num_sessions": args.num_sessions,
            "world_size": world,
            "rank": rank,
        },
        "summary": {
            "total_requests": len(metrics),
            "cache_hits": len(hits),
            "cache_misses": len(misses),
            "hit_rate": len(hits) / len(metrics) if metrics else 0,
            "avg_miss_ttft_ms": float(np.mean([m.e2e_ttft_ms for m in misses])) if misses else 0,
            "avg_hit_ttft_ms": float(np.mean([m.e2e_ttft_ms for m in hits])) if hits else 0,
            "speedup": float(speedup) if hits and misses else 0,
        },
        "per_request": [asdict(m) for m in metrics],
    }

    out_file = results_dir / f"inference_{job_id}_rank{rank}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Cascade Inference Benchmark")
    parser.add_argument("--model", type=str,
                        default="models/Qwen2.5-72B")
    parser.add_argument("--tp-size", type=int, default=None,
                        help="Tensor parallel size (default: auto from model config)")
    parser.add_argument("--dataset", type=str,
                        default="benchmark/data_external/sharegpt/sharegpt_cleaned.json")
    parser.add_argument("--num-sessions", type=int, default=50)
    parser.add_argument("--turns-per-session", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--gpu-capacity", type=float, default=32.0)
    parser.add_argument("--shm-capacity", type=float, default=64.0)
    parser.add_argument("--lustre-path", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="inference_benchmark/results")
    parser.add_argument("--prefix-tokens", type=int, default=0,
                        help="Target system prompt length in tokens (0=short default, 2048=long-context)")
    parser.add_argument("--quantization", type=str, default="none",
                        choices=["none", "int4", "int8"],
                        help="Model quantization (int4=~36GB, int8=~72GB, none=~144GB)")
    parser.add_argument("--no-compression", action="store_true",
                        help="Disable Cascade KV compression")
    args = parser.parse_args()

    run_benchmark(args)

if __name__ == "__main__":
    main()
