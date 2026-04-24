import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

@dataclass
class VLLMRequestMetrics:
    request_id: int
    session_id: int
    turn: int
    cache_hit: bool
    ttft_ms: float
    prefill_tokens: int
    new_tokens: int
    generated_tokens: int

def run_benchmark(args):
    from vllm import LLM, SamplingParams
    from inference_benchmark.workload import ShareGPTWorkload

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log("=" * 70)
    log(f"vLLM Baseline Benchmark")
    log(f"  Model: {args.model}")
    log(f"  APC: {'ON' if args.enable_apc else 'OFF'}")
    log(f"  Sessions: {args.num_sessions}")
    log(f"  Rank: {rank}/{world}")
    log("=" * 70)

    log("\n[1/3] Loading model with vLLM...")
    t0 = time.time()

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="float16",
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=0.90,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        enable_prefix_caching=args.enable_apc,
    )

    load_time = time.time() - t0
    log(f"  Model loaded in {load_time:.1f}s")
    log(f"  APC enabled: {args.enable_apc}")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    log("\n[2/3] Generating workload...")

    tokenizer = llm.get_tokenizer()

    workload = ShareGPTWorkload(args.dataset, tokenizer=tokenizer)
    if args.prefix_tokens > 0:
        workload.set_prefix_length(args.prefix_tokens, tokenizer=tokenizer)
    workload.load(max_conversations=args.num_sessions * 2)
    requests = workload.generate_requests(
        num_sessions=args.num_sessions,
        turns_per_session=args.turns_per_session,
    )

    my_requests = requests[rank::world]
    log(f"  This rank: {len(my_requests)} requests")

    log("\n[3/3] Running inference...")
    metrics: List[VLLMRequestMetrics] = []

    if args.warmup > 0:
        log(f"  Warmup: {args.warmup} requests...")
        for req in my_requests[:args.warmup]:
            _ = llm.generate([req.full_prompt], sampling_params)

    seen_prefixes = set()

    for i, req in enumerate(my_requests):
        is_hit = req.prefix_hash in seen_prefixes
        seen_prefixes.add(req.prefix_hash)

        prefix_tokens = len(tokenizer.encode(req.prefix_prompt))
        total_tokens = len(tokenizer.encode(req.full_prompt))
        new_tokens = total_tokens - prefix_tokens

        max_input = args.max_model_len - args.max_new_tokens
        if total_tokens > max_input:
            continue

        t_start = time.perf_counter()

        outputs = llm.generate([req.full_prompt], sampling_params)

        t_end = time.perf_counter()
        ttft_ms = (t_end - t_start) * 1000

        output = outputs[0]
        num_generated = len(output.outputs[0].token_ids)

        m = VLLMRequestMetrics(
            request_id=req.request_id,
            session_id=req.session_id,
            turn=req.turn,
            cache_hit=is_hit,
            ttft_ms=ttft_ms,
            prefill_tokens=total_tokens,
            new_tokens=new_tokens,
            generated_tokens=num_generated,
        )
        metrics.append(m)

        if i < 5 or i % 20 == 0 or i == len(my_requests) - 1:
            hit_str = "HIT " if is_hit else "MISS"
            log(f"  [{i+1}/{len(my_requests)}] {hit_str} | "
                f"ttft={ttft_ms:.1f}ms, "
                f"prefix={prefix_tokens}, new={new_tokens}")

    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)

    hits = [m for m in metrics if m.cache_hit]
    misses = [m for m in metrics if not m.cache_hit]

    log(f"\nRequests: {len(metrics)} total ({len(hits)} hits, {len(misses)} misses)")
    log(f"Cache hit rate: {len(hits)/len(metrics):.1%}" if metrics else "N/A")
    log(f"APC: {'ON' if args.enable_apc else 'OFF'}")

    if misses:
        miss_ttfts = [m.ttft_ms for m in misses]
        log(f"\nCACHE MISS (full prefill):")
        log(f"  TTFT:  avg={np.mean(miss_ttfts):.1f}ms, "
            f"P50={np.percentile(miss_ttfts, 50):.1f}ms, "
            f"P99={np.percentile(miss_ttfts, 99):.1f}ms")

    if hits:
        hit_ttfts = [m.ttft_ms for m in hits]
        log(f"\nCACHE HIT (APC {'enabled' if args.enable_apc else 'disabled'}):")
        log(f"  TTFT:  avg={np.mean(hit_ttfts):.1f}ms, "
            f"P50={np.percentile(hit_ttfts, 50):.1f}ms, "
            f"P99={np.percentile(hit_ttfts, 99):.1f}ms")

    if hits and misses:
        speedup = np.mean([m.ttft_ms for m in misses]) / np.mean([m.ttft_ms for m in hits])
        log(f"\nSPEEDUP: {speedup:.1f}x (hit vs miss TTFT)")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    apc_tag = "apc_on" if args.enable_apc else "apc_off"
    results = {
        "config": {
            "engine": "vllm",
            "model": args.model,
            "apc": args.enable_apc,
            "tp_size": args.tp_size,
            "num_sessions": args.num_sessions,
            "prefix_tokens": args.prefix_tokens,
            "world_size": world,
            "rank": rank,
        },
        "summary": {
            "total_requests": len(metrics),
            "cache_hits": len(hits),
            "cache_misses": len(misses),
            "hit_rate": len(hits) / len(metrics) if metrics else 0,
            "avg_miss_ttft_ms": float(np.mean([m.ttft_ms for m in misses])) if misses else 0,
            "avg_hit_ttft_ms": float(np.mean([m.ttft_ms for m in hits])) if hits else 0,
            "speedup": float(speedup) if hits and misses else 0,
        },
        "per_request": [asdict(m) for m in metrics],
    }

    out_file = results_dir / f"vllm_{apc_tag}_{job_id}_rank{rank}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="vLLM Baseline Benchmark")
    parser.add_argument("--model", type=str,
                        default="models/Qwen2.5-72B")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dataset", type=str,
                        default="benchmark/data_external/sharegpt/sharegpt_cleaned.json")
    parser.add_argument("--num-sessions", type=int, default=50)
    parser.add_argument("--turns-per-session", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prefix-tokens", type=int, default=0)
    parser.add_argument("--enable-apc", action="store_true",
                        help="Enable vLLM Automatic Prefix Caching")
    parser.add_argument("--results-dir", type=str, default="inference_benchmark/results")
    args = parser.parse_args()

    run_benchmark(args)

if __name__ == "__main__":
    main()
