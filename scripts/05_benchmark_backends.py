"""Benchmark llama.cpp prompt processing and generation speed for the installed llama-cpp-python build.

Uses the same shape as `koboldcpp --benchmark` with a 2048 context: a ~1900 token prompt, then 100 generated tokens.
Run it once per environment / backend build to compare, e.g.:

    uv run --no-sync python scripts/05_benchmark_backends.py models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    UV_PROJECT_ENVIRONMENT=.venv-hip uv run --no-sync python scripts/05_benchmark_backends.py model.gguf

Pass `--cpu` to disable GPU offloading.
"""

import argparse
import time

import llama_cpp
from llama_cpp import Llama


def run(llm: Llama, prompt: list[int], gen_tokens: int) -> tuple[float, float]:
    llm.reset()
    t0 = time.perf_counter()
    llm.eval(prompt)
    t1 = time.perf_counter()
    n = 0
    for _ in llm.generate([], temp=0, reset=False):
        n += 1
        if n == gen_tokens:
            break
    t2 = time.perf_counter()
    return len(prompt) / (t1 - t0), n / (t2 - t1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="path to a GGUF model")
    parser.add_argument("--cpu", action="store_true", help="don't offload any layers to the GPU")
    parser.add_argument("--ctx", type=int, default=2048, help="context size (prompt = ctx - 148 tokens)")
    parser.add_argument("--gen", type=int, default=100, help="tokens to generate")
    parser.add_argument("--runs", type=int, default=2, help="measured runs (best is reported)")
    parser.add_argument("--no-flash-attn", action="store_true")
    args = parser.parse_args()

    llm = Llama(
        args.model,
        n_gpu_layers=0 if args.cpu else -1,
        n_ctx=args.ctx,
        n_batch=512,
        flash_attn=not args.no_flash_attn,
        verbose=False,
        seed=1,
    )
    filler = b"My cat Biscuit is orange and likes sleeping in the sun. " * (args.ctx // 8)
    prompt = llm.tokenize(filler)[: args.ctx - args.gen - 48]

    run(llm, prompt, 8)  # warm-up: kernel / shader compilation for every batch shape
    results = [run(llm, prompt, args.gen) for _ in range(args.runs)]
    pp = max(r[0] for r in results)
    tg = max(r[1] for r in results)
    gpu = "CPU" if args.cpu else ("GPU" if llama_cpp.llama_supports_gpu_offload() else "CPU (no GPU support built)")
    print(f"{args.model}  [{gpu}]")  # noqa: T201
    print(f"prompt processing: {pp:8.0f} tok/s  ({len(prompt)} tokens)")  # noqa: T201
    print(f"generation:        {tg:8.1f} tok/s  ({args.gen} tokens)")  # noqa: T201


if __name__ == "__main__":
    main()
