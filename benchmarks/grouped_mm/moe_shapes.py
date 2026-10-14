"""Time torch._grouped_mm on MoE expert shapes and report % of the HBM roofline.

E=256 experts, hidden H=2048, intermediate I=512, top-8 routing over T tokens. Times the two expert GEMMs,
w13 (N=2I, K=H) and w2 (N=H, K=I), under CUDA-graph replay, after checking them against per-expert matmuls.

    python benchmarks/grouped_mm/moe_shapes.py [--experts 256] [--hbm-tbps 8]
"""

import argparse

import torch


def time_us(fn, iters=100, rounds=5):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    best = float("inf")
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / iters)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inter", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hbm-tbps", type=float, default=8.0)
    args = parser.parse_args()
    E, H, I = args.experts, args.hidden, args.inter
    generator = torch.Generator(device="cuda")
    print(f"{'gemm':>4} {'T':>6} {'rows/expert':>11} {'us':>8} {'roofline':>8}")
    for name, (N, K) in (("w13", (2 * I, H)), ("w2", (H, I))):
        weight = (torch.randn(E, N, K, device="cuda") * 0.02).bfloat16()
        for T in (1, 2, 4, 8, 16, 34, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536):
            generator.manual_seed(T)
            scores = torch.rand(T, E, device="cuda", generator=generator)
            expert_ids = scores.topk(args.top_k, dim=-1).indices.flatten()
            counts = torch.bincount(expert_ids, minlength=E)
            offs = counts.cumsum(0).to(torch.int32)
            M = int(offs[-1])
            activations = torch.randn(M, K, device="cuda").bfloat16()

            out = torch._grouped_mm(activations, weight.transpose(-2, -1), offs=offs)
            per_expert = activations.split(counts.tolist())
            ref = torch.cat([rows.float() @ weight[expert].float().t() for expert, rows in enumerate(per_expert)])
            rel_err = ((out.float() - ref).norm() / ref.norm()).item()
            if rel_err > 1e-2:
                raise RuntimeError(f"{name} T={T}: relative error {rel_err:.2e}")

            us = time_us(lambda: torch._grouped_mm(activations, weight.transpose(-2, -1), offs=offs))
            active_experts = int((counts > 0).sum())
            roofline_us = (active_experts * N * K + M * (K + N)) * 2 / (args.hbm_tbps * 1e6)
            print(f"{name:>4} {T:6d} {M / E:11.1f} {us:8.1f} {100 * roofline_us / us:7.0f}%")


if __name__ == "__main__":
    main()
